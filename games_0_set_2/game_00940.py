
# Generated: 2025-08-27T15:16:30.117521
# Source Brief: brief_00940.md
# Brief Index: 940

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move your yellow square around the grid. "
        "Survive for 60 seconds to advance to the next stage."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric survival game. Dodge the flashing red traps. "
        "Patterns become more complex as you progress through three stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 55)
    COLOR_PLAYER = (255, 220, 0)
    COLOR_PLAYER_GLOW = (255, 220, 0, 50)
    COLOR_TRAP_ACTIVE = (255, 50, 50)
    COLOR_TRAP_INACTIVE = (80, 20, 20)
    COLOR_TRAP_GLOW = (255, 50, 50, 70)
    COLOR_PARTICLE = (255, 120, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_STAGE_BANNER = (0, 0, 0, 150)

    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 17
    GRID_HEIGHT = 11
    TILE_WIDTH_HALF = 18
    TILE_HEIGHT_HALF = 9
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 100

    # Game parameters
    MAX_TOTAL_STEPS = 1800 # 3 stages * 600 steps
    STEPS_PER_STAGE = 600
    TRAP_CYCLE_TIME = 3
    TRAPS_PER_STAGE = {1: 5, 2: 7, 3: 9}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_stage = pygame.font.SysFont("Arial", 32, bold=True)

        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.traps = None
        self.particles = None
        self.stage = None
        self.stage_steps = None
        self.stage_clear_flash = None
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.stage_steps = 0
        self.stage_clear_flash = 0
        
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []
        
        self._spawn_traps()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        reward = 0
        self.steps += 1
        self.stage_steps += 1
        
        # Handle stage clear flash effect
        if self.stage_clear_flash > 0:
            self.stage_clear_flash -= 1

        # Update trap states
        self._update_traps()

        # Handle player movement
        prev_pos = list(self.player_pos)
        moved = self._handle_movement(movement)
        
        # Check for termination
        collision = self._check_collision()
        if collision:
            self.game_over = True
            reward = -100.0
            # sfx: player_death

        # Calculate reward
        if not self.game_over:
            reward += 0.1  # Survival reward

            # Proximity penalty
            if self._is_adjacent_to_active_trap():
                reward -= 0.2
                if moved: # Spawn particles only on risky moves
                    self._spawn_particles(prev_pos, 5, self.COLOR_PARTICLE)

            # Cycle bonus
            if self.steps % self.TRAP_CYCLE_TIME == 0:
                reward += 1.0

            # Stage completion
            if self.stage_steps >= self.STEPS_PER_STAGE:
                reward += 10.0
                self.stage += 1
                self.stage_steps = 0
                self.stage_clear_flash = 20 # Flash for 20 steps
                
                if self.stage > 3:
                    self.game_over = True
                    reward += 100.0 # Win game bonus
                    # sfx: game_win
                else:
                    self._spawn_traps()
                    # sfx: stage_clear

        self.score += max(0, reward)

        terminated = self.game_over or self.steps >= self.MAX_TOTAL_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_traps()
        self._render_player()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
        }

    def _grid_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _spawn_traps(self):
        self.traps = []
        num_traps = self.TRAPS_PER_STAGE[self.stage]
        occupied_positions = {tuple(self.player_pos)}

        for _ in range(num_traps):
            while True:
                pos = [
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT)
                ]
                if tuple(pos) not in occupied_positions:
                    occupied_positions.add(tuple(pos))
                    break
            
            offset = self.np_random.integers(0, self.TRAP_CYCLE_TIME)
            self.traps.append({'pos': pos, 'offset': offset, 'active': False})
        self._update_traps()

    def _update_traps(self):
        for trap in self.traps:
            trap['active'] = (self.steps + trap['offset']) % self.TRAP_CYCLE_TIME == 0

    def _handle_movement(self, movement):
        dx, dy = 0, 0
        if movement == 1: # Up (Iso Up-Left)
            dx, dy = 0, -1
        elif movement == 2: # Down (Iso Down-Right)
            dx, dy = 0, 1
        elif movement == 3: # Left (Iso Down-Left)
            dx, dy = -1, 0
        elif movement == 4: # Right (Iso Up-Right)
            dx, dy = 1, 0
        
        if dx == 0 and dy == 0:
            return False

        new_x = self.player_pos[0] + dx
        new_y = self.player_pos[1] + dy

        if 0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT:
            self.player_pos = [new_x, new_y]
            # sfx: player_move
            return True
        return False

    def _check_collision(self):
        for trap in self.traps:
            if trap['active'] and self.player_pos == trap['pos']:
                return True
        return False

    def _is_adjacent_to_active_trap(self):
        px, py = self.player_pos
        for trap in self.traps:
            if trap['active']:
                tx, ty = trap['pos']
                if abs(px - tx) + abs(py - ty) == 1:
                    return True
        return False

    def _spawn_particles(self, grid_pos, count, color):
        sx, sy = self._grid_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': [sx, sy], 'vel': vel, 'life': life, 'color': color})

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start = self._grid_to_screen(x, 0)
            end = self._grid_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = self._grid_to_screen(0, y)
            end = self._grid_to_screen(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _render_traps(self):
        for trap in self.traps:
            sx, sy = self._grid_to_screen(trap['pos'][0], trap['pos'][1])
            
            if trap['active']:
                color = self.COLOR_TRAP_ACTIVE
                glow_color = self.COLOR_TRAP_GLOW
                size_mod = (math.sin(self.steps * 0.5) + 1) / 2 * 3 # 0 to 3
                
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, self.TILE_WIDTH_HALF, glow_color)
                
                points = [
                    (sx, sy - self.TILE_HEIGHT_HALF + size_mod),
                    (sx - self.TILE_WIDTH_HALF + size_mod, sy + self.TILE_HEIGHT_HALF - size_mod),
                    (sx + self.TILE_WIDTH_HALF - size_mod, sy + self.TILE_HEIGHT_HALF - size_mod)
                ]
            else:
                color = self.COLOR_TRAP_INACTIVE
                points = [
                    (sx, sy - self.TILE_HEIGHT_HALF + 2),
                    (sx - self.TILE_WIDTH_HALF + 2, sy + self.TILE_HEIGHT_HALF - 2),
                    (sx + self.TILE_WIDTH_HALF + 2, sy + self.TILE_HEIGHT_HALF - 2)
                ]

            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_player(self):
        if self.game_over:
            if len(self.particles) < 50:
                self._spawn_particles(self.player_pos, 50, self.COLOR_PLAYER)
            return

        sx, sy = self._grid_to_screen(self.player_pos[0], self.player_pos[1])
        
        glow_radius = int(self.TILE_WIDTH_HALF * 1.5)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, glow_radius, self.COLOR_PLAYER_GLOW)
        
        points = [
            (sx, sy - self.TILE_HEIGHT_HALF),
            (sx + self.TILE_WIDTH_HALF, sy),
            (sx, sy + self.TILE_HEIGHT_HALF),
            (sx - self.TILE_WIDTH_HALF, sy)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.pop(i)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / 20))))
                color = p['color'] + (alpha,)
                radius = int(p['life'] / 5)
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _render_ui(self):
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (15, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        time_left = max(0, (self.STEPS_PER_STAGE - self.stage_steps) / (self.STEPS_PER_STAGE / 60.0))
        timer_text = f"TIME: {time_left:.1f}"
        text_width = self.font_ui.size(timer_text)[0]
        self._draw_text(timer_text, (self.SCREEN_WIDTH - text_width - 15, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        if not self.game_over:
            stage_text = f"STAGE {self.stage}"
            text_width = self.font_ui.size(stage_text)[0]
            self._draw_text(stage_text, (self.SCREEN_WIDTH // 2 - text_width // 2, self.SCREEN_HEIGHT - 30), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        if self.stage_clear_flash > 0:
            text = f"STAGE {self.stage - 1} CLEAR!"
            self._draw_banner_text(text)
        elif self.game_over:
            text = "YOU WIN!" if self.stage > 3 else "GAME OVER"
            self._draw_banner_text(text)

    def _draw_text(self, text, pos, font, color, shadow_color):
        shadow_surface = font.render(text, True, shadow_color)
        text_surface = font.render(text, True, color)
        self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(text_surface, pos)

    def _draw_banner_text(self, text):
        text_surface = self.font_stage.render(text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        banner_rect = text_rect.inflate(40, 20)
        banner_surf = pygame.Surface(banner_rect.size, pygame.SRCALPHA)
        banner_surf.fill(self.COLOR_STAGE_BANNER)
        
        self.screen.blit(banner_surf, banner_rect.topleft)
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    import os
    # Attempt to set a display-compatible driver, fallback to dummy
    try:
        pygame.display.init()
        os.environ["SDL_VIDEODRIVER"] = pygame.display.get_driver()
        use_display = True
    except pygame.error:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        use_display = False

    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)

    if use_display:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Isometric Dodge")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            action = env.action_space.sample()
            action[0] = 0 # Default to no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print(f"Episode finished. Final Info: {info}, Total Reward: {total_reward}")
                total_reward = 0
                env.reset()
                pygame.time.wait(2000)

            clock.tick(10)

        env.close()
    else:
        print("\nNo display detected. Running a short headless test.")
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if (i+1) % 10 == 0:
                print(f"Step {i+1}: Info={info}, Reward={reward:.2f}")
            if terminated:
                print("Episode terminated.")
                env.reset()
        env.close()