
# Generated: 2025-08-28T01:50:05.597802
# Source Brief: brief_04248.md
# Brief Index: 4248

        
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


class Platform:
    """A helper class to manage platform state."""
    def __init__(self, rect, base_y, amplitude, frequency, phase):
        self.rect = rect
        self.base_y = base_y
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def update(self, time_step):
        """Update platform's vertical position based on sinusoidal motion."""
        self.rect.y = self.base_y + self.amplitude * math.sin(self.frequency * time_step + self.phase)

class Particle:
    """A helper class for visual effect particles."""
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        """Update particle position and lifetime."""
        self.pos += self.vel
        self.vel[1] += 0.1  # Gravity on particles
        self.lifetime -= 1
        return self.lifetime > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to aim jump, ↑↓ to adjust power. Press Space to jump."
    )

    game_description = (
        "Leap between moving platforms to reach the top. Avoid falling off the screen. Higher is better!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 180 * FPS # 180 second timer

    # --- Colors ---
    COLOR_BG_BOTTOM = (10, 20, 40)
    COLOR_BG_TOP = (40, 80, 120)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_OUTLINE = (200, 200, 0)
    COLOR_PLATFORM = (60, 120, 255)
    COLOR_PLATFORM_TOP = (150, 200, 255)
    COLOR_GOAL_PLATFORM = (0, 255, 128)
    COLOR_TEXT = (255, 255, 255)
    COLOR_AIM_INDICATOR = (255, 255, 255, 150)

    # --- Physics ---
    GRAVITY = 0.5
    PLAYER_SIZE = 20
    MIN_JUMP_POWER = 5
    MAX_JUMP_POWER = 15
    JUMP_ANGLE_SENSITIVITY = 0.05
    PLAYER_DAMPING = 0.98

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode
        self.screen_array = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)

        self.reset()
        
        # This check is for development and ensures the implementation matches the spec
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT - 50.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_on_ground = True
        self.player_squash = 0.0

        self.jump_power = (self.MIN_JUMP_POWER + self.MAX_JUMP_POWER) / 2
        self.jump_angle = -math.pi / 2

        self.prev_space_held = False

        self.camera_y = 0
        self.max_height_reached = self.player_pos[1]

        self.particles = []
        self._generate_platforms()
        
        # Ensure player starts on the first platform
        start_platform = self.platforms[0]
        self.player_pos[0] = start_platform.rect.centerx
        self.player_pos[1] = start_platform.rect.top - self.PLAYER_SIZE / 2
        self.camera_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.75

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_button, _ = action
        space_held = space_button == 1
        reward = -0.01  # Time penalty

        if not self.game_over:
            self._handle_input(movement, space_held)
            self._update_physics()
            reward += self._check_collisions_and_progress()
            self._update_particles()
            self._update_camera()

        self.steps += 1
        terminated = self._check_termination()
        
        if self.game_over: # Freeze player on game over
            self.player_vel = np.array([0.0, 0.0])

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, False, info
    
    def _handle_input(self, movement, space_held):
        if self.player_on_ground:
            # Adjust angle
            if movement == 3:  # Left
                self.jump_angle -= self.JUMP_ANGLE_SENSITIVITY
            elif movement == 4:  # Right
                self.jump_angle += self.JUMP_ANGLE_SENSITIVITY
            self.jump_angle = np.clip(self.jump_angle, -math.pi + 0.2, -0.2)

            # Adjust power
            if movement == 1:  # Up
                self.jump_power += 0.2
            elif movement == 2:  # Down
                self.jump_power -= 0.2
            self.jump_power = np.clip(self.jump_power, self.MIN_JUMP_POWER, self.MAX_JUMP_POWER)

            # Jump
            if space_held and not self.prev_space_held:
                self.player_vel[0] = self.jump_power * math.cos(self.jump_angle)
                self.player_vel[1] = self.jump_power * math.sin(self.jump_angle)
                self.player_on_ground = False
                self.player_squash = 0.5 # Stretch for jump
                # // SFX: Jump
                self._create_particles(self.player_pos, 15, 'jump')

        self.prev_space_held = space_held
        
    def _update_physics(self):
        # Update platforms
        stage = min(2, self.steps // (60 * self.FPS)) # Difficulty stage
        speed_multiplier = 1.0 + stage * 0.5
        for p in self.platforms:
            p.update(self.steps * speed_multiplier)

        # Update player
        if not self.player_on_ground:
            self.player_vel[1] += self.GRAVITY
            self.player_pos += self.player_vel
            self.player_vel *= self.PLAYER_DAMPING

        # Squash and stretch recovery
        self.player_squash *= 0.85
        
        # Screen bounds
        if self.player_pos[0] < self.PLAYER_SIZE / 2 or self.player_pos[0] > self.SCREEN_WIDTH - self.PLAYER_SIZE / 2:
            self.player_vel[0] *= -1
            self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE/2, self.SCREEN_WIDTH - self.PLAYER_SIZE/2)


    def _check_collisions_and_progress(self):
        reward = 0
        
        # Progress reward/penalty
        height_progress = self.max_height_reached - self.player_pos[1]
        if height_progress > 0:
            reward += 0.1 * height_progress # Reward for upward movement
            self.max_height_reached = self.player_pos[1]
        elif not self.player_on_ground:
            reward += 0.02 * height_progress # Penalty for downward movement (negative progress)

        # Collision with platforms
        if self.player_vel[1] > 0: # Only check for landing if falling
            player_rect = pygame.Rect(
                self.player_pos[0] - self.PLAYER_SIZE/2, 
                self.player_pos[1] - self.PLAYER_SIZE/2, 
                self.PLAYER_SIZE, self.PLAYER_SIZE
            )
            for i, p in enumerate(self.platforms):
                if p.rect.colliderect(player_rect) and player_rect.bottom < p.rect.centery:
                    self.player_on_ground = True
                    self.player_pos[1] = p.rect.top - self.PLAYER_SIZE / 2
                    self.player_vel = np.array([0.0, 0.0])
                    self.player_squash = -0.5 # Squash for landing
                    # // SFX: Land
                    self._create_particles(self.player_pos, 20, 'land')
                    
                    reward += 1.0 # Landing reward
                    self.score += 10
                    
                    if i == len(self.platforms) - 1: # Reached goal
                        self.game_over = True
                        self.win = True
                        reward += 100
                        self.score += 1000
                    break
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        
        reward_penalty = 0
        # Fell off screen
        if self.player_pos[1] > self.camera_y + self.SCREEN_HEIGHT + self.PLAYER_SIZE:
            self.game_over = True
            reward_penalty = -10
            self.score -= 100

        # Max steps reached
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True

        return self.game_over

    def _update_camera(self):
        # Camera follows player upwards
        target_cam_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.4
        self.camera_y = min(self.camera_y, target_cam_y)
    
    def _generate_platforms(self):
        self.platforms = []
        # Start platform
        start_rect = pygame.Rect(self.SCREEN_WIDTH/2 - 50, self.SCREEN_HEIGHT - 20, 100, 20)
        self.platforms.append(Platform(start_rect, start_rect.y, 0, 0, 0))

        current_y = start_rect.y
        
        while current_y > 100:
            y_spacing = self.np_random.uniform(80, 150)
            current_y -= y_spacing
            
            width = self.np_random.uniform(60, 120)
            x_pos = self.np_random.uniform(0, self.SCREEN_WIDTH - width)
            
            amplitude = self.np_random.uniform(5, 30)
            frequency = self.np_random.uniform(0.01, 0.05)
            phase = self.np_random.uniform(0, 2 * math.pi)
            
            rect = pygame.Rect(x_pos, current_y, width, 20)
            self.platforms.append(Platform(rect, current_y, amplitude, frequency, phase))

        # Goal platform
        goal_rect = pygame.Rect(50, 20, self.SCREEN_WIDTH - 100, 30)
        self.platforms.append(Platform(goal_rect, goal_rect.y, 0, 0, 0))

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.SCREEN_HEIGHT - self.max_height_reached,
        }

    def _render_background(self):
        """Draws a vertical gradient."""
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_BOTTOM[0] * (1 - ratio) + self.COLOR_BG_TOP[0] * ratio),
                int(self.COLOR_BG_BOTTOM[1] * (1 - ratio) + self.COLOR_BG_TOP[1] * ratio),
                int(self.COLOR_BG_BOTTOM[2] * (1 - ratio) + self.COLOR_BG_TOP[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        self._render_particles()
        self._render_platforms()
        self._render_player()
        if self.player_on_ground:
            self._render_aim_indicator()
            
    def _render_platforms(self):
        for i, p in enumerate(self.platforms):
            draw_rect = p.rect.copy()
            draw_rect.y -= self.camera_y
            
            is_goal = (i == len(self.platforms) - 1)
            color = self.COLOR_GOAL_PLATFORM if is_goal else self.COLOR_PLATFORM
            
            pygame.draw.rect(self.screen, color, draw_rect, border_radius=3)
            pygame.draw.line(self.screen, self.COLOR_PLATFORM_TOP, 
                             (draw_rect.left + 2, draw_rect.top), 
                             (draw_rect.right - 2, draw_rect.top), 2)

    def _render_player(self):
        # Squash and stretch effect
        squash_x = 1.0 - self.player_squash
        squash_y = 1.0 + self.player_squash
        w = self.PLAYER_SIZE * squash_x
        h = self.PLAYER_SIZE * squash_y
        
        draw_pos_x = self.player_pos[0]
        draw_pos_y = self.player_pos[1] - self.camera_y
        
        player_rect = pygame.Rect(draw_pos_x - w/2, draw_pos_y - h/2, w, h)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, width=2, border_radius=4)

    def _render_aim_indicator(self):
        start_pos = (self.player_pos[0], self.player_pos[1] - self.camera_y)
        length = self.jump_power * 4
        end_pos = (
            start_pos[0] + length * math.cos(self.jump_angle),
            start_pos[1] + length * math.sin(self.jump_angle)
        )
        pygame.draw.aaline(self.screen, self.COLOR_AIM_INDICATOR, start_pos, end_pos, 2)
        pygame.gfxdraw.filled_circle(
            self.screen, int(end_pos[0]), int(end_pos[1]), 4, self.COLOR_AIM_INDICATOR
        )

    def _create_particles(self, pos, count, p_type):
        for _ in range(count):
            if p_type == 'land':
                angle = self.np_random.uniform(math.pi, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                color = random.choice([self.COLOR_PLATFORM, self.COLOR_PLATFORM_TOP, (200,200,200)])
            elif p_type == 'jump':
                angle = self.np_random.uniform(0, math.pi)
                speed = self.np_random.uniform(1, 3)
                color = random.choice([self.COLOR_PLAYER, (255, 255, 150)])
            
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos, vel, radius, color, lifetime))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _render_particles(self):
        for p in self.particles:
            draw_pos = (int(p.pos[0]), int(p.pos[1] - self.camera_y))
            alpha = int(255 * (p.lifetime / p.max_lifetime))
            color = (*p.color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, draw_pos[0], draw_pos[1], int(p.radius), color)
            
    def _render_ui(self):
        height = max(0, int(self.SCREEN_HEIGHT - 20 - self.max_height_reached))
        height_text = self.font_ui.render(f"Height: {height}m", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))

        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH/2 - score_text.get_width()/2, 10))

        time_left = max(0, (self.MAX_EPISODE_STEPS - self.steps) // self.FPS)
        time_text = self.font_ui.render(f"Time: {time_left}s", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_GOAL_PLATFORM if self.win else (255, 50, 50)
            end_text = self.font_game_over.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, 
                                        self.SCREEN_HEIGHT/2 - end_text.get_height()/2))
    
    def close(self):
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Arcade Hopper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # none
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            # Wait for 'R' to reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
                clock.tick(GameEnv.FPS)

        clock.tick(GameEnv.FPS)
        
    env.close()