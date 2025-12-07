import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys for small jumps (← -2, ↓ -1, ↑ +1, → +2). "
        "Use Space for a long forward jump (+3) and Shift for a long backward jump (-3)."
    )

    # Short, user-facing description of the game
    game_description = (
        "A number-line platformer. Leap to the correct target numbers to progress. "
        "Make 15 successful jumps to win, but 3 misses and you're out!"
    )

    # Frames advance only when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_SCALE = 50  # Pixels per number unit

        # Game constants
        self.MAX_MISSES = 3
        self.WIN_JUMPS = 15
        self.MAX_STEPS = 1000

        # Exact spaces as required
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup (headless)
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Define colors and fonts for visual appeal
        self._define_colors_and_fonts()

        # Initialize state variables
        self.player_pos = 0
        self.target_pos = 0
        self.score = 0
        self.misses = 0
        self.successful_jumps = 0
        self.difficulty_tier = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.last_jump_info = None

        # Call reset to ensure a valid initial state
        # self.reset() # reset() is called by the test harness

        # Validate implementation against requirements
        # self.validate_implementation()

    def _define_colors_and_fonts(self):
        """Define the color palette and fonts for the game."""
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_LINE = (70, 80, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_DIM = (150, 150, 170)

        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 150)

        self.COLOR_TARGET = (0, 255, 150)
        self.COLOR_TARGET_GLOW = (0, 150, 80)
        
        self.COLOR_SUCCESS = (50, 255, 150)
        self.COLOR_NEAR_MISS = (255, 200, 50)
        self.COLOR_FAIL = (255, 80, 80)
        
        self.FONT_UI = pygame.font.SysFont("Consolas", 24, bold=True)
        self.FONT_NUMBERS = pygame.font.SysFont("Consolas", 16)
        self.FONT_TARGET = pygame.font.SysFont("Consolas", 20, bold=True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.successful_jumps = 0
        self.difficulty_tier = 0
        self.player_pos = 0
        self.game_over = False
        self.particles = []
        self.last_jump_info = None

        self._generate_new_target()

        return self._get_observation(), self._get_info()

    def _generate_new_target(self):
        """Generates a new target number based on current difficulty."""
        difficulty_range = 2 + self.difficulty_tier
        offset = 0
        while offset == 0:
            offset = self.np_random.integers(-difficulty_range, difficulty_range + 1)
        self.target_pos = self.player_pos + offset

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action and determine jump
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        jump_delta = 0
        if space_held: jump_delta = 3
        elif shift_held: jump_delta = -3
        elif movement == 1: jump_delta = 1  # Up
        elif movement == 2: jump_delta = -1 # Down
        elif movement == 3: jump_delta = -2 # Left
        elif movement == 4: jump_delta = 2  # Right
        
        self.steps += 1
        reward = 0
        self.last_jump_info = None # Clear previous jump visualization

        if jump_delta == 0:
            reward = -0.1 # Small penalty for inaction
        else:
            start_pos = self.player_pos
            land_pos = self.player_pos + jump_delta

            if land_pos == self.target_pos:
                # Successful jump
                reward = 10
                self.score += 10
                self.successful_jumps += 1
                self.player_pos = self.target_pos
                self._spawn_particles(self.player_pos, 'hit')
                # SFX: success_chime.wav
                
                if self.successful_jumps % 3 == 0 and self.successful_jumps > 0:
                    self.difficulty_tier = min(5, self.difficulty_tier + 1)

                if self.successful_jumps < self.WIN_JUMPS:
                    self._generate_new_target()

            else:
                # Missed jump
                error = abs(land_pos - self.target_pos)
                reward = -5
                self.score -= 5
                self.misses += 1
                self.player_pos = land_pos
                particle_type = 'near_miss' if error <= 2 else 'miss'
                self._spawn_particles(self.player_pos, particle_type)
                # SFX: miss_buzz.wav

            self.last_jump_info = {
                "start": start_pos, "end": land_pos, "hit": land_pos == self.target_pos
            }

        terminated = (self.misses >= self.MAX_MISSES) or \
                     (self.successful_jumps >= self.WIN_JUMPS)
        
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            if self.successful_jumps >= self.WIN_JUMPS:
                reward += 50
                self.score += 50
                # SFX: victory_fanfare.wav
            elif self.misses >= self.MAX_MISSES:
                reward -= 20
                self.score -= 20
                # SFX: game_over_sound.wav
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _spawn_particles(self, position, p_type):
        """Spawns a burst of particles at a given position."""
        if p_type == 'hit':
            color = self.COLOR_SUCCESS
            count = 30
        elif p_type == 'near_miss':
            color = self.COLOR_NEAR_MISS
            count = 15
        else: # miss
            color = self.COLOR_FAIL
            count = 15

        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
            lifetime = self.np_random.integers(20, 40)
            # FIX: The particle position must be a 2D list [x, y] to be subscriptable.
            # The 'position' argument is the 1D world x-coordinate. Initialize y to 0.
            self.particles.append([[position, 0], velocity, lifetime, color])

    def _get_observation(self):
        # Clear screen with background color
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the number line, player, target, and effects."""
        center_y = self.HEIGHT // 2 + 50
        
        # Update and draw particles
        self._update_and_draw_particles(center_y)

        # Draw number line and grid
        start_num = self.player_pos - int(self.WIDTH / self.WORLD_SCALE)
        end_num = self.player_pos + int(self.WIDTH / self.WORLD_SCALE)
        
        for i in range(start_num, end_num + 1):
            screen_x = self.WIDTH / 2 + (i - self.player_pos) * self.WORLD_SCALE
            if 0 <= screen_x <= self.WIDTH:
                # Grid lines
                pygame.draw.line(self.screen, self.COLOR_GRID, (int(screen_x), 0), (int(screen_x), self.HEIGHT))
                # Tick marks
                pygame.draw.line(self.screen, self.COLOR_LINE, (int(screen_x), center_y - 5), (int(screen_x), center_y + 5))
                # Numbers
                num_surf = self.FONT_NUMBERS.render(str(i), True, self.COLOR_TEXT_DIM)
                self.screen.blit(num_surf, (int(screen_x - num_surf.get_width() / 2), center_y + 15))
        
        pygame.draw.line(self.screen, self.COLOR_LINE, (0, center_y), (self.WIDTH, center_y), 2)

        # Draw jump arc visualization
        if self.last_jump_info:
            start_x = self.WIDTH / 2 + (self.last_jump_info["start"] - self.player_pos) * self.WORLD_SCALE
            end_x = self.WIDTH / 2 + (self.last_jump_info["end"] - self.player_pos) * self.WORLD_SCALE
            arc_height = -abs(end_x - start_x) * 0.4
            color = self.COLOR_SUCCESS if self.last_jump_info["hit"] else self.COLOR_FAIL
            
            points = []
            steps = 20
            for i in range(steps + 1):
                t = i / steps
                x = start_x + (end_x - start_x) * t
                y = center_y - 25 + 4 * arc_height * t * (1 - t)
                points.append((int(x), int(y)))
            if len(points) > 1:
                pygame.draw.aalines(self.screen, color, False, points, 1)

        # Draw target platform
        if not (self.game_over and self.successful_jumps >= self.WIN_JUMPS):
            target_x = self.WIDTH / 2 + (self.target_pos - self.player_pos) * self.WORLD_SCALE
            platform_rect = pygame.Rect(target_x - 15, center_y - 10, 30, 10)
            pygame.draw.rect(self.screen, self.COLOR_TARGET, platform_rect, border_radius=3)
            pygame.gfxdraw.box(self.screen, platform_rect.inflate(8, 8), (*self.COLOR_TARGET_GLOW, 50))
            
            target_text = self.FONT_TARGET.render(str(self.target_pos), True, self.COLOR_TARGET)
            self.screen.blit(target_text, (int(target_x - target_text.get_width() / 2), center_y - 45))

        # Draw player character
        player_x, player_y = self.WIDTH / 2, center_y - 15
        pygame.gfxdraw.filled_circle(self.screen, int(player_x), int(player_y), 12, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(player_x), int(player_y), 10, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(player_x), int(player_y), 10, self.COLOR_PLAYER)

    def _update_and_draw_particles(self, center_y):
        """Updates physics and draws all active particles."""
        for p in self.particles[:]:
            p[0][0] += p[1][0] * 0.05 # World position update
            p[1][1] += 0.2 # Gravity
            p[0][1] += p[1][1] * 0.05
            p[2] -= 1 # Lifetime

            if p[2] <= 0:
                self.particles.remove(p)
                continue
            
            # Convert world pos to screen pos
            screen_x = self.WIDTH / 2 + (p[0][0] - self.player_pos) * self.WORLD_SCALE
            screen_y = center_y - 10 + p[0][1] # Y is relative to landing spot
            
            if 0 <= screen_x <= self.WIDTH and 0 <= screen_y <= self.HEIGHT:
                size = max(1, int(p[2] / 8))
                alpha = max(0, min(255, int(p[2] * 8)))
                color = (*p[3], alpha)
                try:
                    pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(screen_y), size, color)
                except (OverflowError, ValueError):
                    # Sometimes coordinates can become huge; just skip drawing
                    pass

    def _render_ui(self):
        """Renders the score, misses, and other UI information."""
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Jumps
        jumps_text = self.FONT_UI.render(f"JUMPS: {self.successful_jumps} / {self.WIN_JUMPS}", True, self.COLOR_TEXT)
        self.screen.blit(jumps_text, (self.WIDTH / 2 - jumps_text.get_width() / 2, 10))
        
        # Misses
        miss_text = self.FONT_UI.render("MISSES:", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (self.WIDTH - 150, 10))
        for i in range(self.MAX_MISSES):
            color = self.COLOR_FAIL if i < self.misses else self.COLOR_GRID
            pygame.draw.line(self.screen, color, (self.WIDTH - 40 - i * 20, 15), (self.WIDTH - 25 - i * 20, 35), 3)
            pygame.draw.line(self.screen, color, (self.WIDTH - 40 - i * 20, 35), (self.WIDTH - 25 - i * 20, 15), 3)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.successful_jumps >= self.WIN_JUMPS else "GAME OVER"
            end_font = pygame.font.SysFont("Consolas", 60, bold=True)
            end_text = end_font.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(end_text, (self.WIDTH / 2 - end_text.get_width() / 2, self.HEIGHT / 2 - 50))
            
            reset_font = pygame.font.SysFont("Consolas", 20)
            reset_text = reset_font.render("Call reset() to play again", True, self.COLOR_TEXT)
            self.screen.blit(reset_text, (self.WIDTH / 2 - reset_text.get_width() / 2, self.HEIGHT / 2 + 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "successful_jumps": self.successful_jumps,
            "misses": self.misses,
            "player_pos": self.player_pos,
            "target_pos": self.target_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install gymnasium[classic-control]
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    
    running = True
    game_over_flag = False

    print(env.user_guide)

    while running:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        action_taken = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if game_over_flag:
                    # Any key to reset after game over
                    obs, info = env.reset()
                    game_over_flag = False
                else:
                    action_taken = True
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]: movement = 1
                    elif keys[pygame.K_DOWN]: movement = 2
                    elif keys[pygame.K_LEFT]: movement = 3
                    elif keys[pygame.K_RIGHT]: movement = 4

                    if keys[pygame.K_SPACE]: space = 1
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        if not game_over_flag and action_taken:
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                game_over_flag = True

        # Always render the current state
        frame = np.transpose(env._get_observation(), (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human playability

    env.close()