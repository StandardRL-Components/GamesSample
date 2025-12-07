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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to aim your jump. Hold space to charge and release to jump. "
        "The further you aim, the more powerful the jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade platformer. Hop between procedurally generated platforms, "
        "dodging moving obstacles, to reach the top as quickly as you can."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_HEIGHT = 2000  # Total height of the level

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG_BOTTOM = (20, 10, 40)
        self.COLOR_BG_TOP = (40, 30, 80)
        self.COLOR_PLAYER = (255, 0, 128)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_OBSTACLE = (0, 128, 255)
        self.COLOR_OBSTACLE_OUTLINE = (200, 255, 255)
        self.COLOR_AIM = (255, 255, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PLATFORM_BOTTOM = (0, 255, 128)
        self.COLOR_PLATFORM_TOP = (255, 100, 100)
        self.WIN_PLATFORM_COLOR = (255, 215, 0)

        # Game constants
        self.GRAVITY = 0.3
        self.PLAYER_RADIUS = 10
        self.AIM_SPEED = 4
        self.MAX_AIM_DISTANCE = 80
        self.JUMP_POWER_SCALAR = 0.2
        self.MAX_EPISODE_STEPS = 1500

        # State variables will be initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.aim_offset = None
        self.on_platform_idx = None
        self.is_jumping = None
        self.platforms = None
        self.obstacles = None
        self.particles = None
        self.camera_y = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.max_height_reached = None
        self.obstacle_speed = None
        self.win_platform = None
        self.charge_level = None
        self.prev_space_held = None

        # Initialize state by calling reset, but seed is not available yet.
        # super().reset() will be called first in the actual reset.
        self.np_random = np.random.default_rng()
        self._initialize_state()

    def _initialize_state(self):
        """Initializes all state variables, called from __init__ and reset."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_y = 0
        self.max_height_reached = self.HEIGHT
        self.obstacle_speed = 1.0
        self.charge_level = 0
        self.prev_space_held = False

        self.player_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.aim_offset = pygame.Vector2(0, -30)
        self.on_platform_idx = 0
        self.is_jumping = False
        
        self.particles = []
        self._generate_initial_world()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        step_reward = 0
        self.steps += 1

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused in this design but available

        self._handle_input(movement, space_held)
        step_reward += self._update_state()
        
        self._update_camera()
        self._cull_and_generate_world()
        self._update_difficulty()

        terminated = self.game_over or self.steps >= self.MAX_EPISODE_STEPS
        
        # Final terminal rewards
        if self.game_over:
            if self.player_pos.y > self.HEIGHT + self.camera_y + self.PLAYER_RADIUS:
                step_reward = -100 # Fell off
            elif self.on_platform_idx == -1: # Reached win platform
                step_reward = 100 # Won

        self.score += step_reward

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # --- Aiming ---
        if not self.is_jumping:
            if movement == 1: self.aim_offset.y -= self.AIM_SPEED  # Up
            if movement == 2: self.aim_offset.y += self.AIM_SPEED  # Down
            if movement == 3: self.aim_offset.x -= self.AIM_SPEED  # Left
            if movement == 4: self.aim_offset.x += self.AIM_SPEED  # Right
            
            # Clamp aim distance
            if self.aim_offset.length() > self.MAX_AIM_DISTANCE:
                self.aim_offset.scale_to_length(self.MAX_AIM_DISTANCE)

        # --- Jumping ---
        # The design brief's action mapping is adapted here.
        # Original: Space=medium jump, Shift=high jump
        # Adapted: Space charges the jump, power is based on aim distance.
        if space_held and not self.is_jumping:
            self.charge_level = min(1.0, self.charge_level + 0.1)
        
        # Jump on space release
        if not space_held and self.prev_space_held and not self.is_jumping:
            jump_power = self.aim_offset.length() * self.JUMP_POWER_SCALAR * self.charge_level
            if jump_power > 0.1:
                self.is_jumping = True
                self.player_vel = self.aim_offset.normalize() * jump_power
                self.on_platform_idx = None
                # Sound effect placeholder: # sfx_jump()
                self._create_particles(self.player_pos, (255, 255, 255), 20, 5)
            self.charge_level = 0
        
        self.prev_space_held = space_held

    def _update_state(self):
        reward = 0
        # Update player
        if self.is_jumping:
            reward -= 0.01  # Penalty for being in the air
            self.player_vel.y += self.GRAVITY
            self.player_pos += self.player_vel
        
            # Check for landing on a platform
            for i, platform in enumerate(self.platforms):
                if (platform.left < self.player_pos.x < platform.right and
                    platform.top - self.PLAYER_RADIUS < self.player_pos.y < platform.top and
                    self.player_vel.y > 0):
                    self.is_jumping = False
                    self.player_pos.y = platform.top - self.PLAYER_RADIUS
                    self.player_vel = pygame.Vector2(0, 0)
                    self.on_platform_idx = i
                    reward += 0.5 # Successful jump reward
                    # Sound effect placeholder: # sfx_land()
                    self._create_particles(self.player_pos, self._get_platform_color(platform.y), 15, 3)
                    break
            
            # Check for landing on win platform
            if (self.win_platform.left < self.player_pos.x < self.win_platform.right and
                self.win_platform.top - self.PLAYER_RADIUS < self.player_pos.y < self.win_platform.top and
                self.player_vel.y > 0):
                self.game_over = True
                self.on_platform_idx = -1 # Special index for win
                reward += 10.0 # Big reward for winning
        
        # Check height progress
        current_height_progress = -self.player_pos.y
        max_height_progress = -self.max_height_reached
        if current_height_progress > max_height_progress:
            reward += (current_height_progress - max_height_progress) * 0.01
            self.max_height_reached = self.player_pos.y


        # Update obstacles
        for obstacle in self.obstacles:
            obstacle['rect'].move_ip(obstacle['vel'] * self.obstacle_speed)
            if obstacle['rect'].left < 0 or obstacle['rect'].right > self.WIDTH:
                obstacle['vel'].x *= -1
            if obstacle['rect'].top < -self.WORLD_HEIGHT or obstacle['rect'].bottom > self.HEIGHT:
                obstacle['vel'].y *= -1
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            
        # Check for collisions with obstacles
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_RADIUS, self.player_pos.y - self.PLAYER_RADIUS, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        for obstacle in self.obstacles:
            if player_rect.colliderect(obstacle['rect']):
                self.game_over = True
                reward = -1.0 # Penalty for hitting obstacle
                # Sound effect placeholder: # sfx_hit()
                self._create_particles(self.player_pos, self.COLOR_OBSTACLE, 30, 8)
                self.player_vel.y = 2 # Knockback
                break
        
        # Check for falling off screen
        if self.player_pos.y > self.camera_y + self.HEIGHT + self.PLAYER_RADIUS:
            self.game_over = True

        return reward

    def _update_camera(self):
        # Smoothly follow the player upwards
        target_cam_y = self.player_pos.y - self.HEIGHT * 0.6
        self.camera_y += (target_cam_y - self.camera_y) * 0.05

    def _cull_and_generate_world(self):
        self.platforms = [p for p in self.platforms if p.bottom > self.camera_y - 100]
        self.obstacles = [o for o in self.obstacles if o['rect'].bottom > self.camera_y - 100]

        # Generate new platforms if needed
        highest_platform_y = min(p.y for p in self.platforms) if self.platforms else self.HEIGHT
        if highest_platform_y > self.camera_y - 200:
            self._generate_platforms(highest_platform_y - 50, self.camera_y - 300)

    def _update_difficulty(self):
        if self.steps % 200 == 0 and self.steps > 0:
            self.obstacle_speed = min(3.0, self.obstacle_speed + 0.05)
    
    def _generate_initial_world(self):
        # Start platform
        self.platforms = [pygame.Rect(self.WIDTH//2 - 50, self.HEIGHT - 40, 100, 20)]
        self.obstacles = []
        # Procedurally generate the rest
        self._generate_platforms(self.HEIGHT - 40, -self.WORLD_HEIGHT)
        # Place the win platform at the very top
        self.win_platform = pygame.Rect(self.WIDTH//2 - 60, -self.WORLD_HEIGHT - 50, 120, 30)

    def _generate_platforms(self, y_start, y_end):
        y = y_start
        last_x = self.WIDTH // 2
        while y > y_end:
            dist = self.np_random.uniform(80, 150)
            angle = self.np_random.uniform(math.pi * 0.4, math.pi * 0.6) # Bias upwards
            offset_x = math.cos(angle) * dist
            offset_y = -abs(math.sin(angle) * dist)
            
            new_x = last_x + offset_x
            new_x = np.clip(new_x, 50, self.WIDTH - 50)
            
            width = self.np_random.integers(60, 120)
            
            y += offset_y
            platform = pygame.Rect(new_x - width//2, y, width, 20)
            self.platforms.append(platform)
            last_x = new_x

            # Chance to spawn an obstacle on the new platform
            if self.np_random.random() < 0.4 and y < self.HEIGHT - 200:
                self._generate_obstacle(platform)

    def _generate_obstacle(self, platform):
        size = self.np_random.integers(15, 25)
        rect = pygame.Rect(platform.centerx - size//2, platform.top - size, size, size)
        if self.np_random.random() < 0.5: # Horizontal
            vel = pygame.Vector2(self.np_random.choice([-1, 1]), 0)
        else: # Vertical
            vel = pygame.Vector2(0, self.np_random.choice([-1, 1]))
        self.obstacles.append({'rect': rect, 'vel': vel})

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            vel = pygame.Vector2(self.np_random.uniform(-max_speed, max_speed), self.np_random.uniform(-max_speed, max_speed))
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': 20, 'color': color})

    def _get_platform_color(self, y):
        progress = np.clip((self.HEIGHT - y) / (self.WORLD_HEIGHT + self.HEIGHT), 0, 1)
        return tuple(int(c1 * (1 - progress) + c2 * progress) for c1, c2 in zip(self.COLOR_PLATFORM_BOTTOM, self.COLOR_PLATFORM_TOP))

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw gradient background
        for y in range(self.HEIGHT):
            progress = y / self.HEIGHT
            color = tuple(int(c1 * (1 - progress) + c2 * progress) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Apply camera offset
        def to_screen(pos):
            return int(pos[0]), int(pos[1] - self.camera_y)

        # Draw particles
        for p in self.particles:
            pos_on_screen = to_screen(p['pos'])
            size = max(1, int(p['life'] * 0.2))
            pygame.draw.circle(self.screen, p['color'], pos_on_screen, size)

        # Draw platforms
        for p in self.platforms:
            color = self._get_platform_color(p.y)
            screen_rect = p.move(0, -self.camera_y)
            pygame.draw.rect(self.screen, color, screen_rect, border_radius=5)
        
        # Draw win platform
        screen_rect = self.win_platform.move(0, -self.camera_y)
        pygame.draw.rect(self.screen, self.WIN_PLATFORM_COLOR, screen_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, screen_rect, 2, border_radius=5)


        # Draw obstacles
        for o in self.obstacles:
            screen_rect = o['rect'].move(0, -self.camera_y)
            pygame.gfxdraw.box(self.screen, screen_rect, (*self.COLOR_OBSTACLE, 200))
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, screen_rect, 2, border_radius=3)
        
        # Draw aim indicator and charge bar
        if not self.is_jumping:
            aim_target_pos = self.player_pos + self.aim_offset
            pygame.draw.line(self.screen, self.COLOR_AIM, to_screen(self.player_pos), to_screen(aim_target_pos), 2)
            pygame.gfxdraw.aacircle(self.screen, *to_screen(aim_target_pos), 5, self.COLOR_AIM)
            
            # Charge bar
            if self.charge_level > 0:
                bar_width = 50
                bar_height = 8
                bar_x = self.player_pos.x - bar_width / 2
                bar_y = self.player_pos.y - self.PLAYER_RADIUS - 20
                fill_width = bar_width * self.charge_level
                
                pygame.draw.rect(self.screen, (50, 50, 50), (*to_screen((bar_x, bar_y)), bar_width, bar_height), border_radius=2)
                pygame.draw.rect(self.screen, (255, 255, 0), (*to_screen((bar_x, bar_y)), fill_width, bar_height), border_radius=2)


        # Draw player
        player_screen_pos = to_screen(self.player_pos)
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_screen_pos[0], player_screen_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER_OUTLINE)

        self._render_ui()

    def _render_ui(self):
        # Render score and height
        height_val = max(0, int((self.HEIGHT - 40) - self.player_pos.y))
        height_text = self.font_small.render(f"HEIGHT: {height_val}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))
        
        score_text = self.font_small.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 30))

        # Render time/steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_EPISODE_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Render Game Over message
        if self.game_over:
            if self.on_platform_idx == -1:
                msg = "YOU WIN!"
                color = self.WIN_PLATFORM_COLOR
            else:
                msg = "GAME OVER"
                color = self.COLOR_PLAYER
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": max(0, int((self.HEIGHT - 40) - self.player_pos.y)),
            "is_game_over": self.game_over,
        }
    
    def close(self):
        pygame.quit()


# Example of how to run the environment for visualization
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To run with a human player ---
    # This part requires a screen and is for local testing.
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    pygame.display.set_caption("Arcade Platformer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    running = True
    
    obs, info = env.reset()
    
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            # Wait a bit before resetting on termination
            pygame.time.wait(1000)
            obs, info = env.reset()
            
        env.clock.tick(30) # Run at 30 FPS

    env.close()