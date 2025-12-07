
# Generated: 2025-08-27T13:03:32.115183
# Source Brief: brief_00246.md
# Brief Index: 246

        
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
        "Controls: Use ←→ to aim. Hold [SPACE] to charge your jump, and release to leap. "
        "Reach the top before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An arcade platformer where you leap between procedurally generated platforms. "
        "Master the jump mechanics to ascend quickly and beat the 30-second clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME = 30.0  # seconds
        self.MAX_STEPS = self.MAX_TIME * self.FPS
        self.WIN_HEIGHT = 5000 # Total vertical distance to win
        self.GRAVITY = 0.5
        self.PLAYER_SIZE = 16
        self.JUMP_CHARGE_RATE = 2
        self.MAX_JUMP_CHARGE = 100
        self.MIN_JUMP_POWER = 5
        self.MAX_JUMP_POWER = 18

        # Colors
        self.COLOR_BG_TOP = (20, 30, 80)
        self.COLOR_BG_BOTTOM = (60, 80, 160)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLATFORM = (60, 220, 120)
        self.COLOR_PLATFORM_EDGE = (40, 180, 90)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_CHARGE_BAR = (255, 200, 0)
        self.COLOR_AIM_LINE = (255, 255, 255, 150)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        self._pre_render_background()

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.is_on_platform = None
        self.jump_charge = None
        self.aim_angle = None
        self.platforms = None
        self.camera_y = None
        self.particles = None
        self.last_space_held = None
        self.steps = None
        self.score = None
        self.timer = None
        self.game_over = None
        self.highest_y_pos = None
        self.win_condition_met = None

        # This will call reset()
        self.validate_implementation()


    def _pre_render_background(self):
        """Creates a pre-rendered surface with the background gradient."""
        self.bg_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.WIDTH, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.timer = self.MAX_TIME

        self.player_pos = np.array([self.WIDTH / 2.0, self.HEIGHT - 100.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_on_platform = True
        self.jump_charge = 0
        self.aim_angle = -math.pi / 2  # Straight up
        self.camera_y = 0
        self.particles = []
        self.last_space_held = False

        self.platforms = []
        # Create a large starting platform
        start_platform = pygame.Rect(0, self.HEIGHT - 40, self.WIDTH, 40)
        self.platforms.append(start_platform)
        # Procedurally generate initial platforms
        last_y = start_platform.y
        while last_y > -20:
            y = last_y - self.np_random.integers(60, 120)
            self._generate_new_platform(y)
            last_y = y

        self.highest_y_pos = self.player_pos[1] - self.camera_y

        return self._get_observation(), self._get_info()

    def _generate_new_platform(self, y_pos):
        """Generates a new platform at a given y-coordinate."""
        gap_variance = 150 + (self.steps / 100 * 0.5) # Difficulty scaling
        width = self.np_random.integers(70, 140)
        
        last_plat_x = self.platforms[-1].centerx if self.platforms else self.WIDTH / 2
        min_x = max(20, last_plat_x - gap_variance)
        max_x = min(self.WIDTH - width - 20, last_plat_x + gap_variance)
        
        # Ensure min_x is not greater than max_x
        if min_x >= max_x:
            x = (self.WIDTH - width) / 2
        else:
            x = self.np_random.uniform(min_x, max_x)

        new_platform = pygame.Rect(x, y_pos, width, 20)
        self.platforms.append(new_platform)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Update game logic ---
        self.timer -= 1 / self.FPS
        self.steps += 1

        # Player controls
        if self.is_on_platform:
            # Aiming
            if movement == 3:  # Left
                self.aim_angle -= 0.08
            elif movement == 4:  # Right
                self.aim_angle += 0.08
            self.aim_angle = np.clip(self.aim_angle, -math.pi * 0.9, -math.pi * 0.1)

            # Charging jump
            if space_held:
                self.jump_charge = min(self.MAX_JUMP_CHARGE, self.jump_charge + self.JUMP_CHARGE_RATE)
            
            # Releasing jump
            if not space_held and self.last_space_held and self.jump_charge > 0:
                power = self.MIN_JUMP_POWER + (self.jump_charge / self.MAX_JUMP_CHARGE) * (self.MAX_JUMP_POWER - self.MIN_JUMP_POWER)
                self.player_vel[0] = math.cos(self.aim_angle) * power
                self.player_vel[1] = math.sin(self.aim_angle) * power
                self.is_on_platform = False
                self.jump_charge = 0
                # Sound: Jump

        self.last_space_held = space_held

        # Player physics
        if not self.is_on_platform:
            self.player_vel[1] += self.GRAVITY
            self.player_pos += self.player_vel

            # Wall bouncing
            if self.player_pos[0] < 0 or self.player_pos[0] > self.WIDTH - self.PLAYER_SIZE:
                self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH - self.PLAYER_SIZE)
                self.player_vel[0] *= -0.8 # Dampen horizontal velocity on bounce
                # Sound: Bounce

        # Collision detection with platforms (only when falling)
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        if self.player_vel[1] > 0 and not self.is_on_platform:
            for plat in self.platforms:
                # Check for collision and that the player was above the platform last frame
                if player_rect.colliderect(plat) and (player_rect.bottom - self.player_vel[1]) <= plat.top:
                    self.player_pos[1] = plat.top - self.PLAYER_SIZE
                    self.player_vel = np.array([0.0, 0.0])
                    self.is_on_platform = True
                    reward += 1.0  # Landing reward
                    self.score += 10
                    self._create_landing_particles(player_rect.midbottom)
                    # Sound: Land
                    break

        # Camera scrolling
        scroll_threshold = self.HEIGHT / 2.5
        if self.player_pos[1] < scroll_threshold:
            scroll_amount = scroll_threshold - self.player_pos[1]
            self.camera_y += scroll_amount
            self.player_pos[1] += scroll_amount
            for plat in self.platforms:
                plat.y += scroll_amount
            for p in self.particles:
                p['pos'][1] += scroll_amount
        
        # Platform management
        self.platforms = [p for p in self.platforms if p.top < self.HEIGHT + 50]
        if self.platforms[-1].y > -20:
             self._generate_new_platform(self.platforms[-1].y - self.np_random.integers(80, 150))
        
        # Particle update
        self._update_particles()

        # Calculate rewards
        current_y_pos = self.player_pos[1] - self.camera_y
        if current_y_pos < self.highest_y_pos:
            reward += (self.highest_y_pos - current_y_pos) * 0.01 # Reward for upward movement
            self.score += (self.highest_y_pos - current_y_pos) * 0.01
            self.highest_y_pos = current_y_pos

        if not self.is_on_platform:
            # Penalty for being far from platforms horizontally
            closest_dist_x = min([abs(self.player_pos[0] - p.centerx) for p in self.platforms])
            reward -= (closest_dist_x / self.WIDTH) * 0.1

        # Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.win_condition_met:
                terminal_reward = 100
            else: # Fell, timed out, or max steps
                terminal_reward = -100
            reward += terminal_reward
            self.score += terminal_reward
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.camera_y >= self.WIN_HEIGHT:
            self.win_condition_met = True
            return True
        if self.player_pos[1] > self.HEIGHT:
            return True
        if self.timer <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _create_landing_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifespan': self.np_random.integers(15, 30),
                'color': self.COLOR_PLATFORM,
                'radius': self.np_random.uniform(2, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        # Draw background
        self.screen.blit(self.bg_surface, (0, 0))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]),
                int(p['radius']), (*p['color'], alpha)
            )

        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_EDGE, plat, 3)

        # Draw player and effects
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Player glow
        for i in range(4):
            alpha = 60 - i * 15
            radius = self.PLAYER_SIZE // 2 + i * 3
            pygame.gfxdraw.filled_circle(self.screen, player_rect.centerx, player_rect.centery, radius, (*self.COLOR_PLAYER, alpha))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        if self.is_on_platform:
            # Draw aim line
            aim_end_x = player_rect.centerx + math.cos(self.aim_angle) * (20 + self.jump_charge * 0.5)
            aim_end_y = player_rect.centery + math.sin(self.aim_angle) * (20 + self.jump_charge * 0.5)
            pygame.draw.line(self.screen, self.COLOR_AIM_LINE, player_rect.center, (aim_end_x, aim_end_y), 2)

            # Draw charge bar
            if self.jump_charge > 0:
                bar_width = (self.jump_charge / self.MAX_JUMP_CHARGE) * 40
                bar_rect = pygame.Rect(player_rect.centerx - 20, player_rect.bottom + 5, bar_width, 5)
                pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR, bar_rect)

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Height display
        height_text = f"Height: {int(self.camera_y)} / {self.WIN_HEIGHT}m"
        text_surf = self.font_small.render(height_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Timer display
        timer_text = f"Time: {max(0, self.timer):.1f}s"
        text_surf = self.font_small.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # Score display
        score_text = f"Score: {int(self.score)}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 35))

        # Game over / Win message
        if self.game_over:
            if self.win_condition_met:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            text_surf = self.font_large.render(msg, True, self.COLOR_PLAYER)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "height": self.camera_y,
            "is_on_platform": self.is_on_platform
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11' or 'dummy' for headless, 'windows' for Windows
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Pygame window for human interaction
    pygame.display.set_caption("Leap Platformer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    while running:
        # Get human input
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the observation from the environment
            # Need to transpose back for pygame's display format (W, H)
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()