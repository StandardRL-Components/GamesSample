import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Platform:
    """A helper class to store a pygame.Rect and an associated index."""
    def __init__(self, x, y, w, h, index):
        self.rect = pygame.Rect(x, y, w, h)
        self.index = index

    def __getattr__(self, name):
        """Forward attribute lookups (like .top, .right, .colliderect) to the underlying rect."""
        return getattr(self.rect, name)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to aim your jump. Press space to leap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap across procedurally generated platforms to reach the summit in this fast-paced, side-scrolling arcade hopper."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2500
        self.TARGET_PLATFORM_INDEX = 50
        self.START_LIVES = 3

        # Physics constants
        self.GRAVITY = 0.3
        self.JUMP_POWER_X_SCALE = 0.15
        self.JUMP_POWER_Y_SCALE = 0.25
        self.AIM_SPEED = 2.0
        self.MAX_AIM_RADIUS = 60

        # Color constants
        self.COLOR_BG_TOP = (20, 30, 50)
        self.COLOR_BG_BOTTOM = (40, 60, 100)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLATFORM_FILL = (100, 100, 120)
        self.COLOR_PLATFORM_EDGE = (220, 220, 240)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_HEART = (255, 50, 50)
        self.COLOR_AIMER = (255, 255, 255, 150)
        self.COLOR_PARTICLE = (220, 220, 240)

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 48)

        # Internal state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.player_pos = [0, 0]
        self.player_vel = [0, 0]
        self.player_on_ground = True
        self.player_size = 12
        self.platforms = []
        self.particles = []
        self.camera_x = 0
        self.max_height_reached_y = self.HEIGHT
        self.last_landed_platform_index = 0
        self.platform_difficulty_modifier = 1.0
        self.jump_target_offset = [0, 0]
        self.space_was_pressed = False
        self.last_landed_platform_pos = [0, 0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.START_LIVES
        self.platform_difficulty_modifier = 1.0
        self.last_landed_platform_index = 0
        self.max_height_reached_y = self.HEIGHT

        # Player state
        start_platform = Platform(50, self.HEIGHT - 50, 200, 20, 0)
        self.player_pos = [start_platform.centerx, start_platform.top - self.player_size / 2]
        self.last_landed_platform_pos = list(self.player_pos)
        self.player_vel = [0, 0]
        self.player_on_ground = True
        self.jump_target_offset = [0, -self.MAX_AIM_RADIUS / 2]
        self.space_was_pressed = False

        # World state
        self.platforms = [start_platform]
        self._generate_platforms(20)
        self.camera_x = 0
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if not self.game_over:
            self.steps += 1
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # --- Player Logic ---
            if self.player_on_ground:
                self._handle_aiming(movement)
                self._handle_jump(space_held)
            else:
                self._handle_air_physics()
                reward += self._check_collisions_and_get_reward()

            # --- World & State Updates ---
            self._update_particles()
            self._update_camera()
            self._ensure_platforms_exist()
            reward += self._handle_falling()
            
            # --- Termination Conditions ---
            if self.last_landed_platform_index >= self.TARGET_PLATFORM_INDEX:
                reward += 100
                self.game_over = True
                terminated = True
            
            if self.lives <= 0:
                reward -= 100
                self.game_over = True
                terminated = True

            if self.steps >= self.MAX_STEPS:
                self.game_over = True
                truncated = True # Use truncated for time limit

        self.score += reward
        self.space_was_pressed = action[1] == 1

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Step Sub-functions ---

    def _handle_aiming(self, movement):
        if movement == 1: self.jump_target_offset[1] -= self.AIM_SPEED  # Up
        if movement == 2: self.jump_target_offset[1] += self.AIM_SPEED  # Down
        if movement == 3: self.jump_target_offset[0] -= self.AIM_SPEED  # Left
        if movement == 4: self.jump_target_offset[0] += self.AIM_SPEED  # Right
        
        # Clamp aimer to a circle
        dist = math.hypot(*self.jump_target_offset)
        if dist > self.MAX_AIM_RADIUS:
            self.jump_target_offset[0] = (self.jump_target_offset[0] / dist) * self.MAX_AIM_RADIUS
            self.jump_target_offset[1] = (self.jump_target_offset[1] / dist) * self.MAX_AIM_RADIUS

    def _handle_jump(self, space_held):
        if space_held and not self.space_was_pressed:
            self.player_on_ground = False
            self.player_vel[0] = self.jump_target_offset[0] * self.JUMP_POWER_X_SCALE
            self.player_vel[1] = self.jump_target_offset[1] * self.JUMP_POWER_Y_SCALE

    def _handle_air_physics(self):
        prev_y_vel = self.player_vel[1]
        self.player_vel[1] += self.GRAVITY
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += (prev_y_vel + self.player_vel[1]) * 0.5 # Verlet integration
        
        self.max_height_reached_y = min(self.max_height_reached_y, self.player_pos[1])

    def _check_collisions_and_get_reward(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - self.player_size / 2, self.player_pos[1] - self.player_size / 2, self.player_size, self.player_size)
        
        if self.player_vel[1] > 0: # Only check for landing if moving down
            for p in self.platforms:
                if p.colliderect(player_rect) and player_rect.bottom < p.bottom:
                    # Check if the player's previous position was above the platform
                    prev_bottom = player_rect.bottom - (self.player_vel[1] * 0.5 + (self.player_vel[1] - self.GRAVITY) * 0.5) # Approximate previous position
                    if prev_bottom <= p.top:
                        self.player_on_ground = True
                        self.player_pos[1] = p.top - self.player_size / 2
                        self.player_vel = [0, 0]
                        self.last_landed_platform_pos = list(self.player_pos)
                        self._create_particles(p.midtop, 15)

                        if p.index > self.last_landed_platform_index:
                            reward += 1.0 + (p.index - self.last_landed_platform_index) * 0.1 # Reward for progress
                            # Increase difficulty every 2 new platforms
                            if p.index % 2 == 0 and self.last_landed_platform_index % 2 != 0:
                                self.platform_difficulty_modifier *= 1.05
                            self.last_landed_platform_index = p.index
                        return reward
        return reward

    def _handle_falling(self):
        if self.player_pos[1] > self.HEIGHT + 50:
            self.lives -= 1
            if self.lives > 0:
                self.player_pos = list(self.last_landed_platform_pos)
                self.player_vel = [0, 0]
                self.player_on_ground = True
            return -5.0
        return 0.0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _update_camera(self):
        target_cam_x = self.player_pos[0] - self.WIDTH / 2
        self.camera_x += (target_cam_x - self.camera_x) * 0.1

    def _ensure_platforms_exist(self):
        if self.platforms and self.platforms[-1].right - self.camera_x < self.WIDTH + 200:
            self._generate_platforms(10)

    # --- Generation ---
    def _generate_platforms(self, count):
        last_p = self.platforms[-1]
        for i in range(count):
            gap_x = self.np_random.uniform(40, 120) * self.platform_difficulty_modifier
            new_x = last_p.right + gap_x
            
            delta_y = self.np_random.uniform(-80, 50)
            new_y = np.clip(last_p.y + delta_y, 80, self.HEIGHT - 40)
            
            new_width = self.np_random.uniform(60, 150)
            
            new_platform = Platform(int(new_x), int(new_y), int(new_width), 20, last_p.index + 1)
            self.platforms.append(new_platform)
            last_p = new_platform

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 30),
                'size': self.np_random.uniform(1, 3)
            })

    # --- Rendering ---
    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_BOTTOM)
        top_rect = pygame.Rect(0, 0, self.WIDTH, self.HEIGHT * 0.7)
        pygame.draw.rect(self.screen, self.COLOR_BG_TOP, top_rect)
        
    def _render_game_elements(self):
        cam_x_int = int(self.camera_x)

        # Platforms
        for p in self.platforms:
            if p.right - cam_x_int > 0 and p.left - cam_x_int < self.WIDTH:
                p_rect = p.move(-cam_x_int, 0)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_FILL, p_rect, border_radius=3)
                pygame.draw.line(self.screen, self.COLOR_PLATFORM_EDGE, p_rect.topleft, p_rect.topright, 2)

        # Particles
        for p in self.particles:
            pos_on_screen = (int(p['pos'][0] - cam_x_int), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / 30))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], int(p['size']), (*self.COLOR_PARTICLE, alpha))

        # Player and Aimer
        player_screen_pos = (int(self.player_pos[0] - cam_x_int), int(self.player_pos[1]))
        
        if self.player_on_ground:
            aimer_pos = (player_screen_pos[0] + int(self.jump_target_offset[0]), player_screen_pos[1] + int(self.jump_target_offset[1]))
            pygame.draw.line(self.screen, self.COLOR_AIMER, player_screen_pos, aimer_pos, 1)
            pygame.gfxdraw.filled_circle(self.screen, aimer_pos[0], aimer_pos[1], 4, self.COLOR_AIMER)
            
        player_rect = pygame.Rect(0, 0, self.player_size, self.player_size)
        player_rect.center = player_screen_pos
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        # Lives
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, 20 + i * 25, 20, 8, self.COLOR_UI_HEART)
            pygame.gfxdraw.aacircle(self.screen, 20 + i * 25, 20, 8, self.COLOR_UI_HEART)

        # Height/Score
        height_text = f"Highest: {self.last_landed_platform_index}/{self.TARGET_PLATFORM_INDEX}"
        text_surf = self.font_ui.render(height_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))
        
        # Time/Steps
        steps_text = f"Time: {self.steps // 30:02d}:{self.steps % 30 * 2:02d}"
        text_surf = self.font_ui.render(steps_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.last_landed_platform_index >= self.TARGET_PLATFORM_INDEX:
                msg = "SUMMIT REACHED!"
            else:
                msg = "GAME OVER"
            
            text_surf = self.font_big.render(msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(text_surf, (self.WIDTH//2 - text_surf.get_width()//2, self.HEIGHT//2 - text_surf.get_height()//2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "max_platform": self.last_landed_platform_index,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set `human_mode = False` to run a random agent
    human_mode = True
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real screen for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Hopper")
    
    done = False
    total_reward = 0
    
    action = env.action_space.sample() # Start with a random action
    action.fill(0) # Or reset to no-op

    while not done:
        if human_mode:
            # Human controls
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            
            # Map keys to MultiDiscrete action space
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = np.array([movement, space, shift])

        else: # Random agent
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
    env.close()