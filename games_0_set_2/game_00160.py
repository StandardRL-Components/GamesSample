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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space for a speed boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide your neon robot through a fast-paced, side-scrolling obstacle course. Reach the finish line before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH = 6000
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 20
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_OBSTACLE = (255, 50, 100)
        self.COLOR_FINISH_LINE = (50, 150, 255)
        self.COLOR_UI_TEXT = (240, 240, 255)
        self.COLOR_PARTICLE_BOOST = (100, 200, 255)
        self.COLOR_PARTICLE_TRAIL = (0, 255, 150)
        self.COLOR_PARTICLE_EXPLOSION = [
            (255, 50, 100), (255, 150, 50), (255, 255, 100)
        ]

        # Player properties
        self.PLAYER_SIZE = 20
        self.PLAYER_ACCEL = 1.5
        self.PLAYER_FRICTION = 0.90
        self.PLAYER_MAX_SPEED = 8.0
        
        # Boost properties
        self.BOOST_DURATION_STEPS = 15 # 0.5 seconds
        self.BOOST_COOLDOWN_STEPS = 60 # 2 seconds
        self.BOOST_MULTIPLIER = 2.0

        # Obstacle properties
        self.INITIAL_OBSTACLE_SPEED = 4.0
        self.OBSTACLE_SPEED_INCREASE_INTERVAL = 100 # steps
        self.OBSTACLE_SPEED_INCREASE_AMOUNT = 0.2

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.last_player_world_x = 0
        self.camera_x = 0
        self.obstacle_speed = 0
        self.obstacles = []
        self.particles = []
        self.stars = []
        self.boost_steps_left = 0
        self.boost_cooldown_left = 0
        
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        self.player_pos = pygame.Vector2(100, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        self.camera_x = 0
        self.last_player_world_x = self.player_pos.x
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        
        self.boost_steps_left = 0
        self.boost_cooldown_left = 0
        
        self.particles = []
        self._generate_stars()
        self._generate_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held)
        self._update_player()
        self._update_world()
        self._update_particles()
        
        reward = self._calculate_reward()
        self.score += reward
        
        self._check_collisions()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not truncated: # Game ended due to win/loss, not timeout
            if self.win:
                self.score += 100.0
                reward += 100.0
            else: # Collision
                self.score -= 10.0
                reward -= 10.0
                self._create_explosion(self.player_rect.center)
        
        if terminated:
            self.game_over = True
                
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held):
        accel = self.PLAYER_ACCEL
        max_speed = self.PLAYER_MAX_SPEED

        # Handle boost
        if self.boost_steps_left > 0:
            accel *= self.BOOST_MULTIPLIER
            max_speed *= self.BOOST_MULTIPLIER
            self.boost_steps_left -= 1
            if self.boost_steps_left == 0:
                self.boost_cooldown_left = self.BOOST_COOLDOWN_STEPS
        elif self.boost_cooldown_left > 0:
            self.boost_cooldown_left -= 1
        
        if space_held and self.boost_steps_left == 0 and self.boost_cooldown_left == 0:
            self.boost_steps_left = self.BOOST_DURATION_STEPS
            # Sound: Boost activate
            for _ in range(30):
                self.particles.append(self._create_particle(
                    self.player_rect.center, self.COLOR_PARTICLE_BOOST, 1.5, 20, 
                    angle_range=(-math.pi/8, math.pi/8), speed_mult=3.0
                ))

        # Movement
        if movement == 1: self.player_vel.y -= accel  # Up
        if movement == 2: self.player_vel.y += accel  # Down
        if movement == 3: self.player_vel.x -= accel  # Left
        if movement == 4: self.player_vel.x += accel  # Right

        # Clamp velocity
        if self.player_vel.length() > max_speed:
            self.player_vel.scale_to_length(max_speed)

    def _update_player(self):
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        if self.player_vel.length_squared() < 0.1:
            self.player_vel.update(0, 0)
        
        # Update position
        self.player_pos += self.player_vel

        # Clamp to screen bounds
        self.player_pos.x = max(0, min(self.WIDTH - self.PLAYER_SIZE, self.player_pos.x))
        self.player_pos.y = max(0, min(self.HEIGHT - self.PLAYER_SIZE, self.player_pos.y))
        
        self.player_rect.topleft = self.player_pos

        # Create trail particles
        is_boosting = self.boost_steps_left > 0
        if self.player_vel.length() > 1.0 or is_boosting:
            # Sound: Player whoosh
            p_color = self.COLOR_PARTICLE_BOOST if is_boosting else self.COLOR_PARTICLE_TRAIL
            p_size = 2.0 if is_boosting else 1.0
            p_life = 30 if is_boosting else 20
            
            self.particles.append(self._create_particle(
                self.player_rect.center, p_color, p_size, p_life, 
                base_vel=pygame.Vector2(-self.player_vel.x * 0.5, -self.player_vel.y * 0.5),
                angle_range=(math.pi - 0.5, math.pi + 0.5), speed_mult=0.5
            ))

    def _update_world(self):
        # Update difficulty
        if self.steps > 0 and self.steps % self.OBSTACLE_SPEED_INCREASE_INTERVAL == 0:
            self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE_AMOUNT
        
        self.camera_x += self.obstacle_speed

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _calculate_reward(self):
        current_player_world_x = self.camera_x + self.player_pos.x
        progress = current_player_world_x - self.last_player_world_x
        self.last_player_world_x = current_player_world_x
        
        # Reward for progress, penalize for regress
        if progress > 0:
            return 0.1 * (progress / self.PLAYER_MAX_SPEED)
        else:
            return 0.01 * (progress / self.PLAYER_MAX_SPEED) # progress is negative here

    def _check_collisions(self):
        for obs_rect in self.obstacles:
            # Obstacles are in world coordinates, need to convert to screen coordinates
            screen_obs_rect = obs_rect.move(-self.camera_x, 0)
            if self.player_rect.colliderect(screen_obs_rect):
                self.game_over = True
                self.win = False
                # Sound: Explosion
                return

    def _check_termination(self):
        if self.game_over: # From collision
            return True
        if self.camera_x + self.player_pos.x >= self.WORLD_WIDTH:
            self.win = True
            # Sound: Victory fanfare
            return True
        if self.steps >= self.MAX_STEPS:
            # Sound: Timeout buzzer
            return True # Timeout is handled by truncated flag in step()
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_obstacles()
        self._render_finish_line()
        self._render_particles()
        if not (self.game_over and not self.win): # Don't draw player if they crashed
            self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress_percent": (self.camera_x + self.player_pos.x) / self.WORLD_WIDTH * 100,
            "time_left": max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS)),
        }

    def _render_player(self):
        # Glow effect
        glow_size = self.PLAYER_SIZE * 2.5
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_size/2, glow_size/2), glow_size/2)
        self.screen.blit(glow_surf, (self.player_rect.centerx - glow_size/2, self.player_rect.centery - glow_size/2))
        
        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)

    def _render_obstacles(self):
        for obs_rect in self.obstacles:
            screen_rect = obs_rect.move(-self.camera_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)

    def _render_finish_line(self):
        finish_x = self.WORLD_WIDTH - self.camera_x
        if 0 < finish_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (finish_x, 0), (finish_x, self.HEIGHT), 5)

    def _render_background(self):
        # Parallax stars
        for star in self.stars:
            x = (star['x'] - self.camera_x * star['depth']) % self.WIDTH
            y = star['y']
            size = int(star['size'] * star['depth'])
            if size > 0:
                color_val = int(155 * star['depth'] + 100)
                pygame.draw.rect(self.screen, (color_val, color_val, color_val), (x, y, size, size))

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = int(p['size'] * (p['life'] / p['start_life']))
            
            if size > 0:
                # Use gfxdraw for anti-aliased circles for better look
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p['color'])

    def _render_ui(self):
        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        timer_text = f"TIME: {time_left:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        # Speed
        speed_val = self.obstacle_speed + self.player_vel.x
        speed_text = f"SPEED: {int(speed_val * 10)}"
        speed_surf = self.font_ui.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_surf, (self.WIDTH - speed_surf.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            msg = "FINISH!" if self.win else "GAME OVER"
            color = self.COLOR_FINISH_LINE if self.win else self.COLOR_OBSTACLE
            msg_surf = self.font_game_over.render(msg, True, color)
            self.screen.blit(msg_surf, (self.WIDTH/2 - msg_surf.get_width()/2, self.HEIGHT/2 - msg_surf.get_height()/2))

    def _generate_obstacles(self):
        self.obstacles = []
        current_x = 800
        while current_x < self.WORLD_WIDTH - 500:
            gap_y = self.np_random.integers(0, self.HEIGHT - 150)
            gap_size = self.np_random.integers(120, 180)
            
            # Top obstacle
            top_height = gap_y
            if top_height > 20:
                self.obstacles.append(pygame.Rect(current_x, 0, 50, top_height))
            
            # Bottom obstacle
            bottom_y = gap_y + gap_size
            bottom_height = self.HEIGHT - bottom_y
            if bottom_height > 20:
                self.obstacles.append(pygame.Rect(current_x, bottom_y, 50, bottom_height))
            
            current_x += self.np_random.integers(300, 500)

    def _generate_stars(self):
        self.stars = []
        for _ in range(200):
            depth = self.np_random.uniform(0.1, 1.0)
            self.stars.append({
                'x': self.np_random.integers(0, self.WIDTH),
                'y': self.np_random.integers(0, self.HEIGHT),
                'depth': depth,
                'size': self.np_random.integers(1, 4)
            })

    def _create_particle(self, pos, color, size, life, base_vel=None, angle_range=None, speed_mult=1.0):
        if base_vel is None: base_vel = pygame.Vector2(0,0)
        if angle_range is None: angle_range = (0, 2 * math.pi)

        angle = self.np_random.uniform(angle_range[0], angle_range[1])
        speed = self.np_random.uniform(1.0, 3.0) * speed_mult
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        return {
            'pos': pygame.Vector2(pos),
            'vel': vel + base_vel,
            'color': color,
            'size': size,
            'life': life,
            'start_life': life
        }

    def _create_explosion(self, pos):
        for _ in range(50):
            color = random.choice(self.COLOR_PARTICLE_EXPLOSION)
            self.particles.append(self._create_particle(pos, color, 2.0, 40, speed_mult=2.0))

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Use a separate screen for display if running interactively
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    
    total_reward = 0.0

    while running:
        # Player controls
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
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
                truncated = False
                total_reward = 0.0

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Transpose the observation back for pygame display
            frame_to_show = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame_to_show)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}")
                print(f"Final Info: {info}")
        
    env.close()