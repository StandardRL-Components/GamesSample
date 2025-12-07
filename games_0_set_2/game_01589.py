
# Generated: 2025-08-28T02:04:31.222328
# Source Brief: brief_01589.md
# Brief Index: 1589

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the cavern. Reach the yellow exit tile before time runs out!"
    )

    game_description = (
        "Navigate a procedurally generated 5x5 cavern. Each tile has a unique effect: some are safe, some are traps, and some will teleport you. Find the exit before the 60-second timer expires."
    )

    auto_advance = True

    # --- Constants ---
    # Game
    GRID_SIZE = 5
    TIME_LIMIT_SECONDS = 60
    FPS = 30
    MOVE_COOLDOWN_FRAMES = 8  # Player can move every ~1/4 second

    # Tile Types
    TILE_GREEN = 0  # Safe
    TILE_BLUE = 1   # Teleport
    TILE_RED = 2    # Trap
    TILE_YELLOW = 3 # Exit

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (40, 50, 70)
    COLOR_PLAYER = (220, 255, 255)
    COLOR_TEXT = (230, 230, 230)
    
    TILE_COLORS = {
        TILE_GREEN: (40, 180, 120),
        TILE_BLUE: (80, 150, 255),
        TILE_RED: (220, 70, 90),
        TILE_YELLOW: (255, 200, 50),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game state variables are initialized in reset()
        self.grid = []
        self.player_pos = (0, 0)
        self.player_render_pos = [0, 0]
        self.exit_pos = (0, 0)
        self.time_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.move_cooldown = 0
        self.screen_flash_timer = 0
        self.particles = []

        self.reset()
        
        try:
            self.validate_implementation()
        except AssertionError as e:
            print(f"Implementation validation failed: {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.move_cooldown = 0
        self.screen_flash_timer = 0
        self.particles = []

        self._generate_grid()
        
        # Calculate grid rendering properties
        self.grid_area_size = min(self.width, self.height) * 0.8
        self.cell_size = self.grid_area_size / self.GRID_SIZE
        self.grid_top_left = (
            (self.width - self.grid_area_size) / 2,
            (self.height - self.grid_area_size) / 2
        )
        
        self.player_render_pos = self._get_screen_pos(self.player_pos)

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        # Create a flat list of all possible coordinates
        all_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_coords)

        # Assign special tiles
        self.player_pos = all_coords.pop()
        self.exit_pos = all_coords.pop()
        
        num_red_tiles = 3
        num_blue_tiles = 2
        
        red_tiles = [all_coords.pop() for _ in range(num_red_tiles)]
        blue_tiles = [all_coords.pop() for _ in range(num_blue_tiles)]
        
        # Initialize grid with green tiles
        self.grid = [[self.TILE_GREEN for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        # Place special tiles
        self.grid[self.exit_pos[1]][self.exit_pos[0]] = self.TILE_YELLOW
        for x, y in red_tiles:
            self.grid[y][x] = self.TILE_RED
        for x, y in blue_tiles:
            self.grid[y][x] = self.TILE_BLUE

    def step(self, action):
        reward = -0.01  # Small time penalty per frame
        self.steps += 1
        
        if not self.game_over:
            self.time_remaining -= 1
            if self.move_cooldown > 0:
                self.move_cooldown -= 1
            if self.screen_flash_timer > 0:
                self.screen_flash_timer -= 1
        
        # Unpack action
        movement = action[0]

        if not self.game_over and self.move_cooldown == 0 and movement != 0:
            move_triggered = True
            px, py = self.player_pos
            if movement == 1: # Up
                py -= 1
            elif movement == 2: # Down
                py += 1
            elif movement == 3: # Left
                px -= 1
            elif movement == 4: # Right
                px += 1
            
            if 0 <= px < self.GRID_SIZE and 0 <= py < self.GRID_SIZE:
                self.player_pos = (px, py)
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
                
                # --- Handle Tile Effects and Rewards ---
                tile_type = self.grid[py][px]
                
                if tile_type == self.TILE_GREEN:
                    reward += 1.0 # Arrived at safe tile
                
                elif tile_type == self.TILE_RED:
                    # sfx: trap_triggered
                    reward -= 5.0
                    self.time_remaining -= 5 * self.FPS
                    self.screen_flash_timer = 10 # Flash for 10 frames
                
                elif tile_type == self.TILE_BLUE:
                    # sfx: teleport_whoosh
                    self._create_particles(self._get_screen_pos(self.player_pos), self.TILE_COLORS[self.TILE_BLUE], 50)
                    
                    safe_tiles = []
                    for y_ in range(self.GRID_SIZE):
                        for x_ in range(self.GRID_SIZE):
                            if self.grid[y_][x_] == self.TILE_GREEN and (x_, y_) != self.player_pos:
                                safe_tiles.append((x_, y_))
                    
                    if safe_tiles:
                        self.player_pos = self.np_random.choice(safe_tiles)
                    
                    self._create_particles(self._get_screen_pos(self.player_pos), self.TILE_COLORS[self.TILE_BLUE], 50, inward=True)
                
                elif tile_type == self.TILE_YELLOW:
                    # sfx: level_complete
                    reward += 100.0
                    self.game_over = True

        # Check for time-out termination
        if self.time_remaining <= 0 and not self.game_over:
            # sfx: game_over_sound
            reward -= 100.0
            self.game_over = True
            self.time_remaining = 0
            
        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_screen_pos(self, grid_pos):
        gx, gy = grid_pos
        x = self.grid_top_left[0] + (gx + 0.5) * self.cell_size
        y = self.grid_top_left[1] + (gy + 0.5) * self.cell_size
        return [x, y]

    def _get_observation(self):
        # --- Update render state ---
        target_render_pos = self._get_screen_pos(self.player_pos)
        self.player_render_pos[0] += (target_render_pos[0] - self.player_render_pos[0]) * 0.25
        self.player_render_pos[1] += (target_render_pos[1] - self.player_render_pos[1]) * 0.25
        
        self._update_particles()
        
        # --- Drawing ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and tiles
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                tile_type = self.grid[y][x]
                color = self.TILE_COLORS[tile_type]
                
                rect_x = self.grid_top_left[0] + x * self.cell_size
                rect_y = self.grid_top_left[1] + y * self.cell_size
                tile_rect = pygame.Rect(rect_x, rect_y, self.cell_size, self.cell_size)
                
                pygame.draw.rect(self.screen, color, tile_rect, border_radius=int(self.cell_size * 0.1))
                pygame.draw.rect(self.screen, self.COLOR_GRID, tile_rect, width=2, border_radius=int(self.cell_size * 0.1))

                # Draw exit symbol
                if tile_type == self.TILE_YELLOW:
                    center = tile_rect.center
                    radius = self.cell_size * 0.2
                    pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(radius), (255,255,255, 100))
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1], int(radius), (255,255,255))


        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (p['x'], p['y']), p['size'])

        # Draw player
        px, py = int(self.player_render_pos[0]), int(self.player_render_pos[1])
        player_radius = int(self.cell_size * 0.3)
        for i in range(player_radius, 0, -2):
            alpha = int(150 * (1 - i / player_radius))
            color = (*self.COLOR_PLAYER, alpha)
            pygame.gfxdraw.filled_circle(self.screen, px, py, i, color)
        pygame.gfxdraw.aacircle(self.screen, px, py, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Screen flash for traps
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            alpha = 150 * (self.screen_flash_timer / 10)
            flash_surface.fill((*self.TILE_COLORS[self.TILE_RED], alpha))
            self.screen.blit(flash_surface, (0, 0))

        # Timer
        seconds_left = math.ceil(self.time_remaining / self.FPS)
        timer_text = f"{seconds_left:02d}"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.width - timer_surf.get_width() - 20, 10))

        # Game Over Text
        if self.game_over:
            if self.player_pos == self.exit_pos:
                end_text = "YOU WIN!"
                color = self.TILE_COLORS[self.TILE_YELLOW]
            else:
                end_text = "TIME UP!"
                color = self.TILE_COLORS[self.TILE_RED]
            
            end_surf = self.font_large.render(end_text, True, color)
            end_rect = end_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_surf, end_rect)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] *= 0.98
            p['vy'] *= 0.98
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

    def _create_particles(self, pos, color, count, inward=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            if inward:
                start_x = pos[0] + vx * 15
                start_y = pos[1] + vy * 15
                vx, vy = -vx, -vy
            else:
                start_x, start_y = pos[0], pos[1]
                
            self.particles.append({
                'x': start_x,
                'y': start_y,
                'vx': vx,
                'vy': vy,
                'life': self.np_random.integers(20, 40),
                'size': self.np_random.uniform(2, 5),
                'color': color
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": math.ceil(self.time_remaining / self.FPS)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # Set to True to play manually
    MANUAL_PLAY = True 
    
    if MANUAL_PLAY:
        # --- Manual Control ---
        # This part requires a display
        import os
        # if os.name == 'posix':
        #     os.environ["SDL_VIDEODRIVER"] = "x11"
        # else:
        #     os.environ["SDL_VIDEODRIVER"] = "windows"
        
        live_screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Cavern Puzzle")

        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # Action defaults to no-op
            action = [0, 0, 0] # move, space, shift
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
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

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            live_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(env.FPS)
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}")
                pygame.time.wait(3000) # Wait 3 seconds before closing
                
        env.close()
    
    else:
        # --- RL Agent Simulation ---
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0

        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

        print(f"Episode finished after {step_count} steps.")
        print(f"Total reward: {total_reward}")
        print(f"Final info: {info}")
        env.close()