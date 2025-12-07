
# Generated: 2025-08-27T23:26:54.500071
# Source Brief: brief_03466.md
# Brief Index: 3466

        
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
        "Controls: Use arrow keys to move the miner. Push crystals to create matches of 3 or more."
    )

    game_description = (
        "Isometric puzzle game. Clear all colored crystals by pushing them together before the timer runs out."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 8

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (40, 45, 50, 180)
        self.CRYSTAL_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
        ]
        
        # Isometric projection values
        self.TILE_WIDTH_ISO = 52
        self.TILE_HEIGHT_ISO = 26
        self.ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ORIGIN_Y = 100

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.crystals = None
        self.grid = None
        self.timer = None
        self.max_steps = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.particles = None
        self.initial_crystal_count = None

        self.reset()
        # self.validate_implementation() # Uncomment to run validation check

    def _iso_to_cart(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_WIDTH_ISO / 2
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_HEIGHT_ISO / 2
        return int(screen_x), int(screen_y)

    def _generate_level(self):
        self.crystals = []
        self.grid = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        
        # Place player
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.grid[self.player_pos[1]][self.player_pos[0]] = "P"

        # Generate a guaranteed match
        placed_guarantee = False
        while not placed_guarantee:
            start_x = self.np_random.integers(0, self.GRID_WIDTH - 1)
            start_y = self.np_random.integers(0, self.GRID_HEIGHT - 1)
            color_idx = self.np_random.integers(0, len(self.CRYSTAL_COLORS))
            
            # L-shape
            pos1 = (start_x, start_y)
            pos2 = (start_x + 1, start_y)
            pos3 = (start_x, start_y + 1)
            
            positions = [p for p in [pos1, pos2, pos3] if self._is_valid_and_empty(p)]
            if len(positions) == 3:
                for pos in positions:
                    crystal = {"pos": pos, "color_idx": color_idx, "id": len(self.crystals)}
                    self.crystals.append(crystal)
                    self.grid[pos[1]][pos[0]] = crystal
                placed_guarantee = True

        # Place remaining crystals
        num_crystals = self.np_random.integers(15, 25)
        for _ in range(num_crystals):
            while True:
                x = self.np_random.integers(0, self.GRID_WIDTH)
                y = self.np_random.integers(0, self.GRID_HEIGHT)
                if self._is_valid_and_empty((x, y)):
                    color_idx = self.np_random.integers(0, len(self.CRYSTAL_COLORS))
                    crystal = {"pos": (x, y), "color_idx": color_idx, "id": len(self.crystals)}
                    self.crystals.append(crystal)
                    self.grid[y][x] = crystal
                    break
        
        self.initial_crystal_count = len(self.crystals)

    def _is_valid_and_empty(self, pos):
        x, y = pos
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return False
        if self.grid[y][x] is not None:
            return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.max_steps = 60
        self.timer = self.max_steps
        self.score = 0
        self.game_over = False
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        self.steps += 1
        self.timer -= 1
        reward = -0.01  # Cost per step

        pushed_crystal = False
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            px, py = self.player_pos
            nx, ny = px + dx, py + dy

            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                target_cell = self.grid[ny][nx]
                if target_cell is None: # Move to empty space
                    self.grid[py][px] = None
                    self.player_pos = (nx, ny)
                    self.grid[ny][nx] = "P"
                elif isinstance(target_cell, dict): # Push crystal
                    crystal = target_cell
                    cx, cy = nx + dx, ny + dy
                    
                    pushed_crystal = True
                    # Check if crystal can be pushed
                    if 0 <= cx < self.GRID_WIDTH and 0 <= cy < self.GRID_HEIGHT and self.grid[cy][cx] is None:
                        # Move crystal
                        crystal['pos'] = (cx, cy)
                        self.grid[cy][cx] = crystal
                        # Move player
                        self.grid[py][px] = None
                        self.player_pos = (nx, ny)
                        self.grid[ny][nx] = "P"
                    # Crystal pushed off grid is removed
                    elif not (0 <= cx < self.GRID_WIDTH and 0 <= cy < self.GRID_HEIGHT):
                        self.crystals.remove(crystal)
                        self.grid[ny][nx] = None
                        # Move player
                        self.grid[py][px] = None
                        self.player_pos = (nx, ny)
                        self.grid[ny][nx] = "P"
                        # Placeholder for sound effect
                        # sfx: crystal_shatter_offscreen

        # Check for matches
        match_reward, num_removed = self._check_and_process_matches()
        reward += match_reward
        
        if pushed_crystal and num_removed == 0 and match_reward == 0:
            reward -= 1 # Penalty for inefficient push

        self.score += reward
        
        # Check termination conditions
        win = len(self.crystals) == 0
        loss = self.timer <= 0
        terminated = win or loss
        
        if terminated:
            self.game_over = True
            if win:
                reward += 100 # Win bonus
                # sfx: game_win
            if loss and not win:
                reward -= 50 # Loss penalty
                # sfx: game_lose
            self.score += (100 if win else -50)
            
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_and_process_matches(self):
        if not self.crystals:
            return 0, 0

        reward = 0
        
        q = deque(self.crystals)
        visited_ids = set()
        crystals_to_remove = set()
        
        while q:
            start_crystal = q.popleft()
            if start_crystal['id'] in visited_ids:
                continue

            color_idx = start_crystal['color_idx']
            group = []
            bfs_q = deque([start_crystal])
            group_visited_ids = {start_crystal['id']}

            while bfs_q:
                current_crystal = bfs_q.popleft()
                group.append(current_crystal)
                visited_ids.add(current_crystal['id'])

                cx, cy = current_crystal['pos']
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                        neighbor = self.grid[ny][nx]
                        if isinstance(neighbor, dict) and neighbor['color_idx'] == color_idx and neighbor['id'] not in group_visited_ids:
                            group_visited_ids.add(neighbor['id'])
                            bfs_q.append(neighbor)
            
            if len(group) >= 3:
                reward += 10
                for c in group:
                    crystals_to_remove.add(c['id'])
            elif len(group) == 2:
                reward += 1

        if not crystals_to_remove:
            return reward, 0
        
        # sfx: crystal_match
        num_removed = len(crystals_to_remove)
        
        remaining_crystals = []
        for c in self.crystals:
            if c['id'] in crystals_to_remove:
                self.grid[c['pos'][1]][c['pos'][0]] = None
                self._create_particles(c['pos'], c['color_idx'])
            else:
                remaining_crystals.append(c)
        self.crystals = remaining_crystals
        
        return reward, num_removed

    def _create_particles(self, pos, color_idx):
        cx, cy = self._iso_to_cart(pos[0], pos[1])
        color = self.CRYSTAL_COLORS[color_idx]
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'color': color, 'lifespan': lifespan})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render floor grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._iso_to_cart(x, y)
                points = [
                    (sx, sy),
                    (sx + self.TILE_WIDTH_ISO / 2, sy + self.TILE_HEIGHT_ISO / 2),
                    (sx, sy + self.TILE_HEIGHT_ISO),
                    (sx - self.TILE_WIDTH_ISO / 2, sy + self.TILE_HEIGHT_ISO / 2)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, (50, 60, 70))

        # Render crystals
        for crystal in sorted(self.crystals, key=lambda c: c['pos'][0] + c['pos'][1]):
            self._render_iso_object(crystal['pos'], self.CRYSTAL_COLORS[crystal['color_idx']])

        # Render player
        if self.player_pos:
            self._render_iso_object(self.player_pos, self.COLOR_PLAYER, is_player=True)
            
        # Render particles
        for p in self.particles:
            size = max(1, p['lifespan'] / 6)
            pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _render_iso_object(self, pos, color, is_player=False):
        sx, sy = self._iso_to_cart(pos[0], pos[1])
        depth = 8
        w = self.TILE_WIDTH_ISO * 0.8
        h = self.TILE_HEIGHT_ISO * 1.6
        
        top_points = [
            (sx, sy - h/2 + depth),
            (sx + w/2, sy - h/4 + depth),
            (sx + w/2, sy + h/4 + depth),
            (sx, sy + h/2 + depth),
            (sx - w/2, sy + h/4 + depth),
            (sx - w/2, sy - h/4 + depth),
        ]
        
        darker_color = tuple(max(0, c - 50) for c in color)
        
        # Render sides
        for i in range(3):
            p1 = top_points[i]
            p2 = top_points[i+1]
            p3 = (p2[0], p2[1] - depth)
            p4 = (p1[0], p1[1] - depth)
            pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3,p4], darker_color)

        # Render top face
        top_face_points = [(p[0], p[1] - depth) for p in top_points]
        pygame.gfxdraw.filled_polygon(self.screen, top_face_points, color)
        pygame.gfxdraw.aapolygon(self.screen, top_face_points, tuple(min(255, c + 50) for c in color))
        
        if is_player:
            glow_color = (255, 255, 255, 60)
            glow_surface = pygame.Surface((self.TILE_WIDTH_ISO*2, self.TILE_HEIGHT_ISO*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, glow_color, (self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO), self.TILE_WIDTH_ISO/1.5)
            self.screen.blit(glow_surface, (sx - self.TILE_WIDTH_ISO, sy - self.TILE_HEIGHT_ISO))

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        # Timer
        timer_text = self.font_small.render(f"Time: {self.timer}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (10, 15))

        # Crystals remaining
        crystals_text = self.font_small.render(f"Crystals: {len(self.crystals)} / {self.initial_crystal_count}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystals_text, (self.SCREEN_WIDTH - crystals_text.get_width() - 10, 15))
        
        # Score
        score_text = self.font_small.render(f"Score: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 15))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if len(self.crystals) == 0 else "TIME'S UP!"
            msg_render = self.font_large.render(message, True, (255, 255, 255))
            msg_rect = msg_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_render, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "crystals_left": len(self.crystals)
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a demonstration of how to use the environment
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not terminated:
        action = env.action_space.sample() # Start with a random action
        action[0] = 0 # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        if action[0] != 0: # Only step if a move key was pressed
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
        
        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    print("Game Over!")
    print(f"Final Score: {info['score']:.2f}")
    
    # Keep window open for a bit after game over
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        pygame.display.flip()

    env.close()