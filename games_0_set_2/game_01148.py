
# Generated: 2025-08-27T16:11:28.580346
# Source Brief: brief_01148.md
# Brief Index: 1148

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade platformer where the player must jump between procedurally generated
    platforms to reach the top. The game features minimalist visuals, physics-based
    controls, and a dynamic difficulty curve.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move on platforms or in the air. ↑↓ to adjust jump power. "
        "Press space to jump. Hold shift in the air to fall faster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms to reach the top of the screen in this "
        "minimalist, side-view arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500
    TOTAL_PLATFORMS = 50

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 10)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLATFORM = (150, 150, 170)
    COLOR_PLATFORM_GOAL = (100, 255, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PARTICLE = (200, 200, 255)
    COLOR_AIMER = (255, 100, 100, 150)

    # Physics
    GRAVITY = 0.6
    AIR_RESISTANCE = 0.99
    MOVE_SPEED = 3.5
    AIR_CONTROL = 0.35
    MIN_JUMP_POWER = 12
    MAX_JUMP_POWER = 22
    JUMP_CHARGE_RATE = 0.4
    FAST_FALL_SPEED = 3.0

    # Game Logic
    PLAYER_SIZE = (12, 12)
    PLATFORM_START_WIDTH = 120
    PLATFORM_MIN_WIDTH = 30
    PLATFORM_HEIGHT = 10
    PLATFORM_START_Y_GAP = 80
    PLATFORM_MAX_Y_GAP = 160
    DIFFICULTY_INTERVAL = 5
    WIDTH_DIFFICULTY_SCALAR = 0.95
    GAP_DIFFICULTY_SCALAR = 1.02


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 64)

        # Initialize state variables to be defined in reset()
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.jump_charge = None
        self.current_platform_idx = None
        self.platforms = None
        self.particles = None
        self.camera_y = None
        self.highest_y_pos = None
        self.width_scalar = None
        self.gap_scalar = None
        self.steps = None
        self.score = None
        self.start_time = None
        self.game_over = None
        self.win = None
        self.np_random = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.start_time = pygame.time.get_ticks()

        self.camera_y = 0.0
        self.width_scalar = 1.0
        self.gap_scalar = 1.0
        self.platforms = self._generate_platforms()
        
        start_platform = self.platforms[0]
        self.player_pos = np.array([start_platform.centerx, start_platform.top - self.PLAYER_SIZE[1]], dtype=float)
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = True
        self.current_platform_idx = 0
        self.jump_charge = self.MIN_JUMP_POWER
        self.highest_y_pos = self.player_pos[1]

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = self._update_game_state(action)
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100.0  # Goal-oriented reward for winning
            else:
                reward -= 100.0  # Goal-oriented reward for losing
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, action):
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        step_reward = 0.0

        # --- Handle Input and State Changes ---
        if self.on_ground:
            # Horizontal movement on platform
            if movement == 3: self.player_pos[0] -= self.MOVE_SPEED
            if movement == 4: self.player_pos[0] += self.MOVE_SPEED
            
            # Clamp player to platform edges
            current_plat = self.platforms[self.current_platform_idx]
            self.player_pos[0] = np.clip(self.player_pos[0], current_plat.left, current_plat.right)

            # Charge jump
            if movement == 1: self.jump_charge = min(self.MAX_JUMP_POWER, self.jump_charge + self.JUMP_CHARGE_RATE)
            if movement == 2: self.jump_charge = max(self.MIN_JUMP_POWER, self.jump_charge - self.JUMP_CHARGE_RATE)

            # Execute jump
            if space_pressed:
                self.player_vel[1] = -self.jump_charge
                # Add slight horizontal velocity based on position relative to platform center
                self.player_vel[0] = (self.player_pos[0] - current_plat.centerx) * 0.1
                self.on_ground = False
                # Sound: Jump

        # --- Physics Update ---
        if not self.on_ground:
            # Air control
            if movement == 3: self.player_vel[0] -= self.AIR_CONTROL
            if movement == 4: self.player_vel[0] += self.AIR_CONTROL
            
            # Fast fall
            if shift_held:
                self.player_vel[1] = max(self.player_vel[1], self.FAST_FALL_SPEED)

            # Apply gravity and air resistance
            self.player_vel[1] += self.GRAVITY
            self.player_vel[0] *= self.AIR_RESISTANCE
            self.player_pos += self.player_vel

        # --- Continuous Rewards ---
        height_gain = self.highest_y_pos - self.player_pos[1]
        if height_gain > 0:
            step_reward += height_gain * 0.1
            self.highest_y_pos = self.player_pos[1]
        
        if self.on_ground and self.current_platform_idx < len(self.platforms) - 1:
            next_platform = self.platforms[self.current_platform_idx + 1]
            dist_to_center = abs(self.player_pos[0] - next_platform.centerx)
            step_reward -= dist_to_center * 0.001

        # --- Collision Detection ---
        if self.player_vel[1] > 0 and not self.on_ground:
            player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE[0] / 2, self.player_pos[1], *self.PLAYER_SIZE)
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat) and self.player_pos[1] < plat.bottom:
                    if i > self.current_platform_idx:
                        step_reward += 1.0  # Event-based reward for landing on a new platform
                        self.score += 1
                        if self.score > 0 and self.score % self.DIFFICULTY_INTERVAL == 0:
                            self.width_scalar *= self.WIDTH_DIFFICULTY_SCALAR
                            self.gap_scalar *= self.GAP_DIFFICULTY_SCALAR
                        self.current_platform_idx = i
                        if self.current_platform_idx == self.TOTAL_PLATFORMS - 1:
                            self.win = True
                    
                    self.on_ground = True
                    self.player_vel = np.array([0.0, 0.0])
                    self.player_pos[1] = plat.top - self.PLAYER_SIZE[1]
                    self.jump_charge = self.MIN_JUMP_POWER
                    self._spawn_particles(self.player_pos + np.array([0, self.PLAYER_SIZE[1]]))
                    # Sound: Land
                    break
        
        # --- Camera and Particle Update ---
        self._update_camera()
        self._update_particles()
        
        return step_reward

    def _check_termination(self):
        # Fell off screen or reached max steps or won
        return (self.player_pos[1] > self.camera_y + self.SCREEN_HEIGHT + 50 or
                self.steps >= self.MAX_STEPS or
                self.win)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000.0 if not self.game_over else self.elapsed_time
        if self.game_over and self.elapsed_time is None:
             self.elapsed_time = elapsed_time

        return {
            "score": self.score,
            "steps": self.steps,
            "time": round(elapsed_time, 2),
            "height": -int(self.highest_y_pos)
        }

    def _generate_platforms(self):
        platforms = []
        last_x = self.SCREEN_WIDTH / 2
        current_y = self.SCREEN_HEIGHT - 50

        for i in range(self.TOTAL_PLATFORMS):
            width = max(self.PLATFORM_MIN_WIDTH, self.PLATFORM_START_WIDTH * self.width_scalar)
            if i == 0:
                width = self.SCREEN_WIDTH * 0.8
            
            y_gap = self.np_random.uniform(self.PLATFORM_START_Y_GAP, self.PLATFORM_MAX_Y_GAP) * self.gap_scalar
            current_y -= y_gap
            
            max_x_offset = self.SCREEN_WIDTH / 2 - width / 2
            min_x_reach = last_x - 200 # Max horizontal jump distance approx
            max_x_reach = last_x + 200
            
            x_pos = self.np_random.uniform(
                max(width / 2, min_x_reach),
                min(self.SCREEN_WIDTH - width / 2, max_x_reach)
            )
            
            last_x = x_pos
            platforms.append(pygame.Rect(x_pos - width / 2, current_y, width, self.PLATFORM_HEIGHT))

        return platforms

    def _update_camera(self):
        # Smoothly follow player upwards
        target_cam_y = self.player_pos[1] - self.SCREEN_HEIGHT / 2.5
        self.camera_y += (target_cam_y - self.camera_y) * 0.1

    def _spawn_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(20, 40)
            self.particles.append([pos.copy(), vel, life])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1]  # Update position
            p[1][1] += self.GRAVITY * 0.1  # Particles have slight gravity
            p[2] -= 1  # Reduce life
        self.particles = [p for p in self.particles if p[2] > 0]

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw particles
        for pos, vel, life in self.particles:
            screen_pos = (int(pos[0]), int(pos[1] - self.camera_y))
            if 0 <= screen_pos[0] < self.SCREEN_WIDTH and 0 <= screen_pos[1] < self.SCREEN_HEIGHT:
                alpha = max(0, min(255, int(255 * (life / 30.0))))
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], 2, (*self.COLOR_PARTICLE, alpha))

        # Draw platforms
        for i, plat in enumerate(self.platforms):
            screen_rect = plat.move(0, -self.camera_y)
            color = self.COLOR_PLATFORM_GOAL if i == self.TOTAL_PLATFORMS - 1 else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, screen_rect, border_radius=3)
        
        # Draw player
        player_screen_pos = (self.player_pos[0], self.player_pos[1] - self.camera_y)
        player_rect = pygame.Rect(
            player_screen_pos[0] - self.PLAYER_SIZE[0] / 2,
            player_screen_pos[1] - self.PLAYER_SIZE[1],
            *self.PLAYER_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_BG_BOTTOM, player_rect, width=1, border_radius=2)

        # Draw jump aimer
        if self.on_ground:
            aim_length = self.jump_charge * 3
            aim_angle = -math.pi/2
            end_pos = (
                player_rect.centerx + aim_length * math.cos(aim_angle),
                player_rect.centery + aim_length * math.sin(aim_angle)
            )
            pygame.draw.aaline(self.screen, self.COLOR_AIMER, player_rect.center, end_pos, 2)

    def _render_ui(self):
        info = self._get_info()
        
        # Height display
        height_text = f"Height: {info['height']}"
        text_surf = self.font_ui.render(height_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Score display
        score_text = f"Platforms: {self.score}/{self.TOTAL_PLATFORMS-1}"
        text_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 30))

        # Time display
        time_text = f"Time: {info['time']:.2f}s"
        text_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

        # Game Over message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLATFORM_GOAL if self.win else self.COLOR_AIMER
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, (*self.COLOR_BG_BOTTOM, 200), text_rect.inflate(20, 20), border_radius=10)
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to initialize states for observation
        self.reset()
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset return values
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame window setup for human play ---
    pygame.display.set_caption("Icy Ascent")
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop for human interaction
    while not done:
        # Action mapping from keyboard to MultiDiscrete
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # For auto_advance=True, this controls the speed

    env.close()