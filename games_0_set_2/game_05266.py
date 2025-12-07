
# Generated: 2025-08-28T04:28:25.773146
# Source Brief: brief_05266.md
# Brief Index: 5266

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move selector. Space to select/deselect a crystal. "
        "Hold Shift and use arrows to push a selected crystal."
    )

    game_description = (
        "An isometric puzzle game. Push crystals onto gems to collect them. "
        "You have a limited number of pushes. Plan your moves carefully!"
    )

    auto_advance = False

    # --- Constants ---
    COLOR_BG = (26, 33, 54)
    COLOR_WALL = (52, 73, 94)
    COLOR_WALL_TOP = (65, 91, 117)
    COLOR_FLOOR = (44, 62, 80)
    COLOR_GRID = (55, 75, 95)
    COLOR_GEM = (241, 196, 15)
    COLOR_CURSOR = (255, 255, 255)
    CRYSTAL_COLORS = [
        (231, 76, 60), (52, 152, 219), (46, 204, 113),
        (155, 89, 182), (243, 156, 18)
    ]
    TEXT_COLOR = (236, 240, 241)
    TEXT_SHADOW_COLOR = (40, 40, 40)

    GRID_WIDTH = 14
    GRID_HEIGHT = 10
    NUM_CRYSTALS = 8
    NUM_GEMS = 10
    MAX_MOVES = 20
    
    TILE_W = 50
    TILE_H = 26
    CRYSTAL_H = 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        self.origin_x = self.screen_width // 2
        self.origin_y = 100

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.crystals = []
        self.gems = []
        self.particles = deque()

        self.reset()

        self.validate_implementation()

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.TILE_W / 2
        screen_y = self.origin_y + (x + y) * self.TILE_H / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_rect(self, surface, color, x, y, h):
        sx, sy = self._iso_to_screen(x, y)
        points = [
            (sx, sy - h),
            (sx + self.TILE_W / 2, sy + self.TILE_H / 2 - h),
            (sx, sy + self.TILE_H - h),
            (sx - self.TILE_W / 2, sy + self.TILE_H / 2 - h)
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_iso_cube(self, surface, color, x, y, h):
        sx, sy = self._iso_to_screen(x, y)
        top_color = tuple(min(255, c + 30) for c in color)
        side_color_right = color
        side_color_left = tuple(max(0, c - 20) for c in color)

        # Top face
        top_points = [
            (sx, sy - h), (sx + self.TILE_W / 2, sy + self.TILE_H / 2 - h),
            (sx, sy + self.TILE_H - h), (sx - self.TILE_W / 2, sy + self.TILE_H / 2 - h)
        ]
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)

        # Right face
        right_points = [
            (sx, sy + self.TILE_H - h), (sx + self.TILE_W / 2, sy + self.TILE_H / 2 - h),
            (sx + self.TILE_W / 2, sy + self.TILE_H / 2), (sx, sy + self.TILE_H)
        ]
        pygame.gfxdraw.filled_polygon(surface, right_points, side_color_right)

        # Left face
        left_points = [
            (sx, sy + self.TILE_H - h), (sx - self.TILE_W / 2, sy + self.TILE_H / 2 - h),
            (sx - self.TILE_W / 2, sy + self.TILE_H / 2), (sx, sy + self.TILE_H)
        ]
        pygame.gfxdraw.filled_polygon(surface, left_points, side_color_left)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.gems_collected_count = 0
        self.game_over = False
        self.win = False

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_crystal_idx = None
        self.last_space_held = False
        
        self.particles.clear()
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid.fill(0)
        self.grid[0, :] = 1; self.grid[-1, :] = 1
        self.grid[:, 0] = 1; self.grid[:, -1] = 1

        available_coords = []
        for r in range(1, self.GRID_WIDTH - 1):
            for c in range(1, self.GRID_HEIGHT - 1):
                available_coords.append([r, c])
        
        self.np_random.shuffle(available_coords)
        
        occupied_coords = set()

        self.crystals = []
        for i in range(self.NUM_CRYSTALS):
            pos = available_coords.pop(0)
            self.crystals.append({
                'pos': pos,
                'color': self.CRYSTAL_COLORS[i % len(self.CRYSTAL_COLORS)],
                'id': i
            })
            occupied_coords.add(tuple(pos))

        self.gems = []
        for i in range(self.NUM_GEMS):
            pos = available_coords.pop(0)
            self.gems.append({'pos': pos, 'active': True, 'id': i})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.1
        
        push_executed = False

        if self.selected_crystal_idx is not None and shift_held and movement > 0:
            push_reward = self._handle_push(movement)
            reward += push_reward
            self.moves_left -= 1
            push_executed = True
        elif space_held and not self.last_space_held:
            self._handle_selection()
        elif not push_executed and movement > 0:
            self._move_cursor(movement)
        
        self.last_space_held = space_held
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 50
                self.score += 50
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_push(self, direction):
        # sfx: push_crystal.wav
        crystal = self.crystals[self.selected_crystal_idx]
        start_pos = list(crystal['pos'])
        
        delta = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}[direction]
        
        current_pos = list(start_pos)
        while True:
            next_pos = [current_pos[0] + delta[0], current_pos[1] + delta[1]]
            
            if self.grid[next_pos[0], next_pos[1]] == 1:
                break
            
            is_occupied = False
            for c in self.crystals:
                if c['pos'] == next_pos:
                    is_occupied = True
                    break
            if is_occupied:
                break
                
            current_pos = next_pos
        
        end_pos = current_pos
        crystal['pos'] = end_pos
        self.selected_crystal_idx = None
        
        # Particle trail
        path_len = abs(end_pos[0] - start_pos[0]) + abs(end_pos[1] - start_pos[1])
        for i in range(path_len * 3):
            t = i / (path_len * 3)
            p_pos = [start_pos[0] + (end_pos[0] - start_pos[0]) * t, 
                     start_pos[1] + (end_pos[1] - start_pos[1]) * t]
            self._create_particles(p_pos, crystal['color'], 1, 15, 0.5)

        # Check for gem collection
        push_reward = 0
        for gem in self.gems:
            if gem['active'] and gem['pos'] == end_pos:
                # sfx: collect_gem.wav
                gem['active'] = False
                self.gems_collected_count += 1
                self.score += 1
                push_reward += 1
                self._create_particles(end_pos, self.COLOR_GEM, 20, 30, 1.5)
                break
        
        return push_reward

    def _handle_selection(self):
        # sfx: select.wav
        if self.selected_crystal_idx is not None:
            self.selected_crystal_idx = None
        else:
            for i, crystal in enumerate(self.crystals):
                if crystal['pos'] == self.cursor_pos:
                    self.selected_crystal_idx = i
                    break

    def _move_cursor(self, direction):
        # sfx: cursor_move.wav
        delta = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}[direction]
        new_pos = [self.cursor_pos[0] + delta[0], self.cursor_pos[1] + delta[1]]
        
        if 0 < new_pos[0] < self.GRID_WIDTH -1 and 0 < new_pos[1] < self.GRID_HEIGHT -1:
            self.cursor_pos = new_pos

    def _check_termination(self):
        if self.gems_collected_count >= self.NUM_GEMS:
            self.game_over = True
            self.win = True
            return True
        if self.moves_left <= 0:
            self.game_over = True
            return True
        return False

    def _create_particles(self, pos, color, count, lifetime, speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'color': color,
                'life': lifetime,
                'max_life': lifetime,
            })

    def _update_and_draw_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'][0] += p['vel'][0] * 0.1
            p['pos'][1] += p['vel'][1] * 0.1
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.popleft()
            else:
                sx, sy = self._iso_to_screen(p['pos'][0], p['pos'][1])
                alpha = 255 * (p['life'] / p['max_life'])
                size = 3 * (p['life'] / p['max_life'])
                r, g, b = p['color']
                pygame.draw.circle(self.screen, (r, g, b, alpha), (sx, sy), int(size))
    
    def _draw_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.TEXT_SHADOW_COLOR)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Game Elements ---
        render_queue = []
        
        # Floor and Walls
        for r in range(self.GRID_WIDTH):
            for c in range(self.GRID_HEIGHT):
                render_queue.append({'type': 'tile', 'pos': [r, c], 'is_wall': self.grid[r, c] == 1})
        
        # Gems
        for gem in self.gems:
            if gem['active']:
                render_queue.append({'type': 'gem', 'pos': gem['pos'], 'id': gem['id']})

        # Crystals
        for i, crystal in enumerate(self.crystals):
            render_queue.append({'type': 'crystal', 'pos': crystal['pos'], 'color': crystal['color'], 'idx': i})

        # Sort by depth for correct isometric rendering
        render_queue.sort(key=lambda item: (item['pos'][0] + item['pos'][1], item.get('type') != 'tile'))

        for item in render_queue:
            x, y = item['pos']
            if item['type'] == 'tile':
                color = self.COLOR_WALL_TOP if item['is_wall'] else self.COLOR_FLOOR
                h = 10 if item['is_wall'] else 0
                self._draw_iso_rect(self.screen, color, x, y, h)
                if not item['is_wall']:
                    sx, sy = self._iso_to_screen(x,y)
                    points = [(sx, sy), (sx + self.TILE_W / 2, sy + self.TILE_H / 2),
                              (sx, sy + self.TILE_H), (sx - self.TILE_W / 2, sy + self.TILE_H / 2)]
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

            elif item['type'] == 'gem':
                pulse = (math.sin(self.steps * 0.1 + item['id']) + 1) / 2
                size = 8 + pulse * 4
                sx, sy = self._iso_to_screen(x, y)
                pygame.draw.circle(self.screen, self.COLOR_GEM, (sx, sy + 5), int(size))
                pygame.draw.circle(self.screen, (255,255,255), (sx, sy + 5), int(size*0.5))

            elif item['type'] == 'crystal':
                is_selected = self.selected_crystal_idx == item['idx']
                if is_selected:
                    sx, sy = self._iso_to_screen(x, y)
                    pulse = (math.sin(self.steps * 0.2) + 1) / 2
                    color = (255, 255, 255, 100 + pulse * 100)
                    pygame.gfxdraw.filled_circle(self.screen, sx, sy + self.TILE_H // 2, int(self.TILE_W/2), color)
                self._draw_iso_cube(self.screen, item['color'], x, y, self.CRYSTAL_H)
        
        # Cursor
        cx, cy = self.cursor_pos
        sx, sy = self._iso_to_screen(cx, cy)
        cursor_color = (*self.COLOR_CURSOR, 100)
        cursor_points = [
            (sx, sy), (sx + self.TILE_W / 2, sy + self.TILE_H / 2),
            (sx, sy + self.TILE_H), (sx - self.TILE_W / 2, sy + self.TILE_H / 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, cursor_points, cursor_color)
        pygame.gfxdraw.aapolygon(self.screen, cursor_points, (*self.COLOR_CURSOR, 200))

        # Particles
        self._update_and_draw_particles()

        # --- Render UI ---
        self._draw_text(f"Moves: {self.moves_left}", self.font_medium, self.TEXT_COLOR, (20, 20))
        self._draw_text(f"Gems: {self.gems_collected_count}/{self.NUM_GEMS}", self.font_medium, self.TEXT_COLOR, (self.screen_width - 160, 20))

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "OUT OF MOVES"
            self._draw_text(msg, self.font_large, self.TEXT_COLOR, (self.screen_width/2 - self.font_large.size(msg)[0]/2, self.screen_height/2 - 40))
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "gems_collected": self.gems_collected_count,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    pygame.display.set_caption("Crystal Caverns")
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    running = True
    while running:
        space_pressed = False
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                 space_pressed = True

        # --- Action Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # For turn-based, only register one action per key press event
        # This is a simplified manual control scheme
        current_action = [movement, 1 if space_pressed else 0, 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0]
        
        # Only step if an action is taken
        if any(current_action):
            obs, reward, terminated, truncated, info = env.step(np.array(current_action))
            done = terminated or truncated
            print(f"Action: {current_action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Limit frame rate for manual play

    env.close()