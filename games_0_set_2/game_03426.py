
# Generated: 2025-08-27T23:19:52.627389
# Source Brief: brief_03426.md
# Brief Index: 3426

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to swap crystals, Space to cycle selection. Match 3 or more to score!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Isometric puzzle game. Swap crystals to create matches of three or more. Clear the board before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 8
    TILE_W_HALF, TILE_H_HALF = 28, 14
    
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_LINE = (40, 60, 80)
    COLOR_GRID_BASE = (30, 50, 70)
    CRYSTAL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_WHITE = (240, 240, 240)
    
    # Game States
    STATE_IDLE = "IDLE"
    STATE_SWAP_ANIM = "SWAP_ANIM"
    STATE_MATCH_CHECK = "MATCH_CHECK"
    STATE_CLEAR_ANIM = "CLEAR_ANIM"
    STATE_FALL_ANIM = "FALL_ANIM"
    
    # Animation Timings (in frames)
    ANIM_DURATION_SWAP = 6
    ANIM_DURATION_CLEAR = 10
    ANIM_DURATION_FALL = 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_H_HALF) // 2 + 20

        self.reset()
        
        # self.validate_implementation() # Uncomment for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_board()
        
        self.selected_pos = (0, 0)
        
        self.game_state = self.STATE_IDLE
        self.animation_timer = 0
        self.pending_swap = None
        self.is_swap_back = False
        self.turn_reward = 0.0
        
        self.particles = []
        self.last_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        terminated = False
        
        self._update_animations()

        if self.game_state == self.STATE_IDLE:
            # Handle player input only when idle
            space_pressed = space_held and not self.last_space_held
            
            if space_pressed:
                # sound: select_crystal.wav
                self._cycle_selection()
            
            if movement != 0:
                # 1=up, 2=down, 3=left, 4=right
                dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
                x1, y1 = self.selected_pos
                x2, y2 = x1 + dx, y1 + dy

                if 0 <= x2 < self.GRID_WIDTH and 0 <= y2 < self.GRID_HEIGHT:
                    self.pending_swap = ((x1, y1), (x2, y2))
                    self._start_swap_anim(is_swap_back=False)
                    self.game_state = self.STATE_SWAP_ANIM
                else:
                    reward = -0.1 # Penalty for invalid move attempt
        
        self.last_space_held = space_held
        
        # If the turn is resolved, add accumulated reward
        if self.game_state == self.STATE_IDLE and self.turn_reward != 0:
            reward += self.turn_reward
            self.turn_reward = 0

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 50 # Time out penalty
        
        if self._is_board_clear():
            terminated = True
            reward += 100 # Board clear bonus
            
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_animations(self):
        # --- State Machine for animations and game logic flow ---
        if self.animation_timer > 0:
            self.animation_timer -= 1
        
        if self.game_state == self.STATE_SWAP_ANIM and self.animation_timer == 0:
            self.game_state = self.STATE_MATCH_CHECK
        
        elif self.game_state == self.STATE_MATCH_CHECK:
            (x1, y1), (x2, y2) = self.pending_swap
            self._swap_grid_logic((x1, y1), (x2, y2)) # Apply swap to logic grid
            
            matches = self._find_all_matches()
            if not matches:
                if not self.is_swap_back:
                    # Invalid move, swap back
                    self._start_swap_anim(is_swap_back=True)
                    self.game_state = self.STATE_SWAP_ANIM
                    self.turn_reward = -0.1 # Penalty for a move that creates no match
                else:
                    # Swapped back, turn is over
                    self.game_state = self.STATE_IDLE
            else:
                # Matches found!
                # sound: match_found.wav
                self._handle_matches(matches)
                self.game_state = self.STATE_CLEAR_ANIM
                self.animation_timer = self.ANIM_DURATION_CLEAR

        elif self.game_state == self.STATE_CLEAR_ANIM and self.animation_timer == 0:
            self._apply_gravity()
            self.game_state = self.STATE_FALL_ANIM
            self.animation_timer = self.ANIM_DURATION_FALL
        
        elif self.game_state == self.STATE_FALL_ANIM and self.animation_timer == 0:
            # After falling, check for new matches (cascades)
            matches = self._find_all_matches()
            if matches:
                # sound: cascade_match.wav
                self._handle_matches(matches)
                self.game_state = self.STATE_CLEAR_ANIM
                self.animation_timer = self.ANIM_DURATION_CLEAR
            else:
                # No more matches, turn is over
                self.game_state = self.STATE_IDLE
        
        # Update particle physics
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Game Logic ---
    def _generate_board(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                # Avoid creating matches during generation
                possible_colors = list(range(1, len(self.CRYSTAL_COLORS) + 1))
                if x > 1 and self.grid[x-1, y] == self.grid[x-2, y]:
                    if self.grid[x-1, y] in possible_colors:
                        possible_colors.remove(self.grid[x-1, y])
                if y > 1 and self.grid[x, y-1] == self.grid[x, y-2]:
                     if self.grid[x, y-1] in possible_colors:
                        possible_colors.remove(self.grid[x, y-1])
                self.grid[x, y] = self.np_random.choice(possible_colors)
        
        # Create visual representation tied to the grid
        self.visual_grid = []
        for x in range(self.GRID_WIDTH):
            col = []
            for y in range(self.GRID_HEIGHT):
                col.append(self._create_crystal_vis(x, y))
            self.visual_grid.append(col)

    def _create_crystal_vis(self, x, y, color_idx=None):
        if color_idx is None:
            color_idx = self.grid[x, y]
        screen_x, screen_y = self._grid_to_screen(x, y)
        return {
            'pos': [screen_x, screen_y],
            'start_pos': [screen_x, screen_y],
            'target_pos': [screen_x, screen_y],
            'color_idx': color_idx,
            'size_mult': 1.0,
            'alpha': 255
        }

    def _cycle_selection(self):
        x, y = self.selected_pos
        x += 1
        if x >= self.GRID_WIDTH:
            x = 0
            y = (y + 1) % self.GRID_HEIGHT
        self.selected_pos = (x, y)

    def _start_swap_anim(self, is_swap_back):
        self.is_swap_back = is_swap_back
        (x1, y1), (x2, y2) = self.pending_swap
        
        vis1 = self.visual_grid[x1][y1]
        vis2 = self.visual_grid[x2][y2]

        vis1['start_pos'] = list(vis1['pos'])
        vis1['target_pos'] = list(vis2['pos'])
        vis2['start_pos'] = list(vis2['pos'])
        vis2['target_pos'] = list(vis1['pos'])
        
        self.animation_timer = self.ANIM_DURATION_SWAP

    def _swap_grid_logic(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
        self.visual_grid[x1][y1], self.visual_grid[x2][y2] = self.visual_grid[x2][y2], self.visual_grid[x1][y1]

    def _find_all_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[x, y] != 0 and self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[x, y] != 0 and self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return list(matches)

    def _handle_matches(self, matches):
        num_cleared = len(matches)
        self.turn_reward += num_cleared * 1.0 # +1 per crystal
        if num_cleared > 3:
            self.turn_reward += 5.0 # Combo bonus
        self.score += self.turn_reward

        for x, y in matches:
            color_idx = self.grid[x, y]
            if color_idx > 0:
                self.visual_grid[x][y]['size_mult'] = 1.5 # Start clear animation
                self._spawn_particles(x, y, color_idx)
            self.grid[x, y] = 0 # Mark for removal

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    # Move crystal down
                    self.grid[x, y + empty_count] = self.grid[x, y]
                    self.grid[x, y] = 0
                    
                    # Update visual representation
                    vis = self.visual_grid[x][y]
                    self.visual_grid[x][y + empty_count] = vis
                    
                    sx, sy = self._grid_to_screen(x, y + empty_count)
                    vis['start_pos'] = list(vis['pos'])
                    vis['target_pos'] = [sx, sy]
            
            # Fill new crystals at the top
            for i in range(empty_count):
                new_color = self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1)
                self.grid[x, i] = new_color
                
                sx, sy = self._grid_to_screen(x, i)
                start_y = self._grid_to_screen(x, i - empty_count)[1]
                
                vis = self._create_crystal_vis(x, i)
                vis['pos'] = [sx, start_y]
                vis['start_pos'] = [sx, start_y]
                vis['target_pos'] = [sx, sy]
                self.visual_grid[x][i] = vis

    def _is_board_clear(self):
        return np.all(self.grid == 0)

    def _spawn_particles(self, grid_x, grid_y, color_idx):
        screen_x, screen_y = self._grid_to_screen(grid_x, grid_y)
        color = self.CRYSTAL_COLORS[color_idx-1]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [screen_x, screen_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                'life': random.randint(15, 30),
                'color': color,
                'radius': random.uniform(1, 3)
            })

    # --- Rendering ---
    def _grid_to_screen(self, x, y):
        screen_x = self.grid_origin_x + (x - y) * self.TILE_W_HALF
        screen_y = self.grid_origin_y + (x + y) * self.TILE_H_HALF
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw grid tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self._draw_iso_tile(self.screen, x, y)
        
        # Draw selection highlight
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        color = (200 + 55 * pulse, 200 + 55 * pulse, 220 + 35 * pulse)
        self._draw_iso_tile(self.screen, self.selected_pos[0], self.selected_pos[1], color, border_width=2)

        # Draw crystals
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x,y] > 0:
                    self._draw_crystal(self.screen, self.visual_grid[x][y])
                    
        # Draw particles
        for p in self.particles:
            life_ratio = p['life'] / 30.0
            p_color = (p['color'][0] * life_ratio, p['color'][1] * life_ratio, p['color'][2] * life_ratio)
            pygame.draw.circle(self.screen, p_color, p['pos'], p['radius'])

    def _draw_iso_tile(self, surface, x, y, color=None, border_width=0):
        sx, sy = self._grid_to_screen(x, y)
        points = [
            (sx, sy - self.TILE_H_HALF),
            (sx + self.TILE_W_HALF, sy),
            (sx, sy + self.TILE_H_HALF),
            (sx - self.TILE_W_HALF, sy)
        ]
        
        # Draw base for 3D effect
        base_points = [(p[0], p[1] + 4) for p in points]
        pygame.gfxdraw.filled_polygon(surface, base_points, self.COLOR_GRID_BASE)
        
        if color:
            pygame.gfxdraw.aapolygon(surface, points, color)
            if border_width > 1: # Draw thicker border if needed
                pygame.draw.polygon(surface, color, points, border_width)
        else:
            pygame.gfxdraw.aapolygon(surface, points, self.COLOR_GRID_LINE)
    
    def _draw_crystal(self, surface, vis_data):
        anim_progress = 1.0
        if self.game_state in [self.STATE_SWAP_ANIM, self.STATE_FALL_ANIM] and self.animation_timer > 0:
            duration = self.ANIM_DURATION_SWAP if self.game_state == self.STATE_SWAP_ANIM else self.ANIM_DURATION_FALL
            anim_progress = 1.0 - (self.animation_timer / duration)
        
        # Interpolate position
        x = vis_data['start_pos'][0] + (vis_data['target_pos'][0] - vis_data['start_pos'][0]) * anim_progress
        y = vis_data['start_pos'][1] + (vis_data['target_pos'][1] - vis_data['start_pos'][1]) * anim_progress
        vis_data['pos'] = [x, y]

        # Handle clear animation
        if self.game_state == self.STATE_CLEAR_ANIM and self.animation_timer > 0:
            progress = self.animation_timer / self.ANIM_DURATION_CLEAR
            vis_data['size_mult'] = 1.0 + (1.0 - progress) * 0.7
            vis_data['alpha'] = progress * 255
        else:
            vis_data['size_mult'] = 1.0
            vis_data['alpha'] = 255

        color_idx = vis_data['color_idx']
        if color_idx == 0: return

        base_color = self.CRYSTAL_COLORS[color_idx-1]
        highlight_color = tuple(min(255, c + 60) for c in base_color)
        
        radius = int(self.TILE_H_HALF * 0.9 * vis_data['size_mult'])
        
        # Use a temporary surface for alpha blending
        temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        
        pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*base_color, vis_data['alpha']))
        pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, (*base_color, vis_data['alpha']))
        
        # Highlight
        h_radius = int(radius * 0.5)
        pygame.gfxdraw.filled_circle(temp_surf, radius, radius, h_radius, (*highlight_color, vis_data['alpha']))

        surface.blit(temp_surf, (int(x - radius), int(y - radius)))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        secs = (time_left // 30) % 60
        total_secs = time_left // 30
        time_str = f"Time: {total_secs:02d}"
        time_text = self.font_main.render(time_str, True, self.COLOR_WHITE)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            
            end_text_str = "Board Cleared!" if self._is_board_clear() else "Time's Up!"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # env.validate_implementation() # Run self-checks
    
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for display
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)

        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space and Shift
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}")
        
        if terminated:
            print("Game Over!")
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Match the environment's intended FPS

    env.close()