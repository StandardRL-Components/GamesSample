
# Generated: 2025-08-27T20:16:09.261867
# Source Brief: brief_02404.md
# Brief Index: 2404

        
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
        "Controls: Arrow keys to move your blue square. Push the brown crates onto the green targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced Sokoban puzzle. Race against the clock to push all crates onto their targets."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    TILE_SIZE = 40
    GRID_PIXEL_WIDTH = GRID_SIZE * TILE_SIZE
    GRID_PIXEL_HEIGHT = GRID_SIZE * TILE_SIZE
    GRID_OFFSET = ((SCREEN_WIDTH - GRID_PIXEL_WIDTH) // 2, (SCREEN_HEIGHT - GRID_PIXEL_HEIGHT) // 2)

    MAX_TIME = 60  # seconds
    MAX_STEPS = 1800 # 60 seconds * 30 fps
    FPS = 30

    COLOR_BG = (25, 28, 32)
    COLOR_WALL = (60, 65, 70)
    COLOR_FLOOR = (40, 44, 52)
    COLOR_TARGET = (80, 120, 90)
    COLOR_TARGET_FILLED = (120, 180, 135)
    COLOR_PLAYER = (80, 170, 255)
    COLOR_PLAYER_GLOW = (80, 170, 255, 50)
    COLOR_CRATE = (160, 110, 80)
    COLOR_CRATE_BORDER = (120, 80, 60)
    
    LEVELS = [
        [
            "WWWWWWWWWW",
            "W        W",
            "W   C T  W",
            "W   P    W",
            "W        W",
            "W        W",
            "W        W",
            "W        W",
            "W        W",
            "WWWWWWWWWW",
        ],
        [
            "WWWWWWWWWW",
            "W T      W",
            "W C WWWW W",
            "W   W  C W",
            "W P W  T W",
            "W   W    W",
            "W C WWWW W",
            "W T      W",
            "W        W",
            "WWWWWWWWWW",
        ],
        [
            "WWWWWWWWWW",
            "W P      W",
            "W C WWWW W",
            "W   W  C W",
            "W   W  T W",
            "W   W    W",
            "W C WWWW W",
            "W T      W",
            "W   T    W",
            "WWWWWWWWWW",
        ],
        [
            "WWWWWWWWWW",
            "WT       W",
            "WC  C   TW",
            "W   P    W",
            "W C    C W",
            "W T    T W",
            "W        W",
            "W C    C W",
            "WT       W",
            "WWWWWWWWWW",
        ],
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.timer_font = pygame.font.SysFont("monospace", 30, bold=True)
        self.info_font = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.end_font = pygame.font.SysFont("sans-serif", 50, bold=True)
        
        # Initialize state variables
        self.player_pos = (0, 0)
        self.crates = []
        self.targets = []
        self.walls = []
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.win_state = False
        self.particles = []
        
        self.reset()
        
        # self.validate_implementation() # Optional validation call
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_TIME
        self.game_over = False
        self.win_state = False
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def _generate_level(self):
        level_idx = self.np_random.integers(0, len(self.LEVELS))
        level_data = self.LEVELS[level_idx]

        self.crates.clear()
        self.targets.clear()
        self.walls.clear()

        for r, row_str in enumerate(level_data):
            for c, char in enumerate(row_str):
                pos = (c, r)
                if char == 'P':
                    self.player_pos = pos
                elif char == 'C':
                    self.crates.append(pos)
                elif char == 'T':
                    self.targets.append(pos)
                elif char == 'W':
                    self.walls.append(pos)

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            
            # --- Game Logic ---
            reward += self._handle_movement(movement)
            
            # Update time
            self.time_remaining -= 1 / self.FPS
            
            # Update particles
            self._update_particles()
            
            # --- Check Termination ---
            on_target_count = sum(1 for c in self.crates if c in self.targets)
            all_on_target = on_target_count == len(self.targets)
            
            if all_on_target:
                reward += 50  # Win bonus
                self.game_over = True
                self.win_state = True
                terminated = True
                # sfx_win
            elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
                reward -= 50 # Loss penalty
                self.time_remaining = 0
                self.game_over = True
                self.win_state = False
                terminated = True
                # sfx_lose

        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        reward = 0
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        else: return 0 # No-op

        old_player_pos = self.player_pos
        new_player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
        
        old_crate_positions = list(self.crates)
        crates_moved = False

        # Check for wall collision
        if new_player_pos in self.walls:
            return 0 # sfx_bump_wall

        # Check for crate collision (push)
        if new_player_pos in self.crates:
            crate_index = self.crates.index(new_player_pos)
            new_crate_pos = (new_player_pos[0] + dx, new_player_pos[1] + dy)
            
            # Check if crate can be pushed
            if new_crate_pos not in self.walls and new_crate_pos not in self.crates:
                self.player_pos = new_player_pos
                self.crates[crate_index] = new_crate_pos
                crates_moved = True
                self._add_push_particles(old_player_pos, new_player_pos)
                # sfx_push_crate
            else:
                return 0 # Can't push crate
        else: # Move to empty space
            self.player_pos = new_player_pos

        # --- Calculate Rewards ---
        # 1. Distance-based reward
        if crates_moved:
            old_distances = [min(self._manhattan_distance(c, t) for t in self.targets) for c in old_crate_positions]
            new_distances = [min(self._manhattan_distance(c, t) for t in self.targets) for c in self.crates]
            
            for i in range(len(self.crates)):
                dist_diff = old_distances[i] - new_distances[i]
                if dist_diff > 0:
                    reward += 0.1  # Moved closer
                elif dist_diff < 0:
                    reward -= 0.02 # Moved further

        # 2. Event-based reward for placing a crate on a target
        crates_on_target_before = sum(1 for c in old_crate_positions if c in self.targets)
        crates_on_target_after = sum(1 for c in self.crates if c in self.targets)
        if crates_on_target_after > crates_on_target_before:
            reward += 1.0 * (crates_on_target_after - crates_on_target_before)
            # sfx_crate_on_target
            
        return reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw floor
        floor_rect = pygame.Rect(self.GRID_OFFSET[0], self.GRID_OFFSET[1], self.GRID_PIXEL_WIDTH, self.GRID_PIXEL_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, floor_rect)

        # Draw targets
        for tx, ty in self.targets:
            center_x = self.GRID_OFFSET[0] + int((tx + 0.5) * self.TILE_SIZE)
            center_y = self.GRID_OFFSET[1] + int((ty + 0.5) * self.TILE_SIZE)
            radius = int(self.TILE_SIZE * 0.35)
            color = self.COLOR_TARGET_FILLED if (tx, ty) in self.crates else self.COLOR_TARGET
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)

        # Draw walls
        for wx, wy in self.walls:
            rect = pygame.Rect(self.GRID_OFFSET[0] + wx * self.TILE_SIZE, self.GRID_OFFSET[1] + wy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            
        # Draw particles
        self._render_particles()

        # Draw crates
        for cx, cy in self.crates:
            rect = pygame.Rect(self.GRID_OFFSET[0] + cx * self.TILE_SIZE, self.GRID_OFFSET[1] + cy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, self.COLOR_CRATE_BORDER, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_CRATE, inner_rect, border_radius=3)

        # Draw player
        px, py = self.player_pos
        player_center_x = self.GRID_OFFSET[0] + int((px + 0.5) * self.TILE_SIZE)
        player_center_y = self.GRID_OFFSET[1] + int((py + 0.5) * self.TILE_SIZE)
        
        # Glow effect
        glow_radius = int(self.TILE_SIZE * 0.5)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(temp_surf, (player_center_x - glow_radius, player_center_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Player square
        player_rect = pygame.Rect(0, 0, self.TILE_SIZE - 6, self.TILE_SIZE - 6)
        player_rect.center = (player_center_x, player_center_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        
    def _render_ui(self):
        # --- Timer Display ---
        time_text = f"{int(self.time_remaining // 60):02}:{int(self.time_remaining % 60):02}"
        time_color = (255, 80, 80) if self.time_remaining < 10 and not self.game_over else (220, 220, 220)
        time_surf = self.timer_font.render(time_text, True, time_color)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(time_surf, time_rect)

        # --- Crate Counter Display ---
        on_target_count = sum(1 for c in self.crates if c in self.targets)
        total_crates = len(self.crates)
        info_text = f"CRATES: {on_target_count} / {total_crates}"
        info_color = (180, 220, 180) if on_target_count == total_crates else (220, 220, 220)
        info_surf = self.info_font.render(info_text, True, info_color)
        info_rect = info_surf.get_rect(bottomleft=(15, self.SCREEN_HEIGHT - 10))
        self.screen.blit(info_surf, info_rect)

        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text = "LEVEL COMPLETE" if self.win_state else "TIME UP"
            end_color = (150, 255, 150) if self.win_state else (255, 150, 150)
            end_surf = self.end_font.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "crates_on_target": sum(1 for c in self.crates if c in self.targets),
            "total_crates": len(self.crates)
        }
        
    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _add_push_particles(self, from_pos, to_pos):
        px = self.GRID_OFFSET[0] + (to_pos[0] + 0.5) * self.TILE_SIZE
        py = self.GRID_OFFSET[1] + (to_pos[1] + 0.5) * self.TILE_SIZE
        push_dir = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        
        for _ in range(8):
            angle = math.atan2(-push_dir[1], -push_dir[0]) + self.np_random.uniform(-math.pi / 2, math.pi / 2)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            size = self.np_random.uniform(2, 4)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'size': size})
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color = (*self.COLOR_CRATE, alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'] * (p['life'] / 20.0))
            if size > 0:
                rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
                pygame.draw.rect(self.screen, color, rect)

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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a separate display for human play
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Keyboard input mapping
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        if keys[pygame.K_r]: # Press 'r' to reset
            obs, info = env.reset()
            total_reward = 0
            done = False
            continue

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    env.close()
    pygame.quit()