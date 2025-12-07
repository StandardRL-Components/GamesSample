
# Generated: 2025-08-28T05:26:57.272140
# Source Brief: brief_02627.md
# Brief Index: 2627

        
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

    user_guide = (
        "Controls: ↑/↓ to move vertically. Hold SPACE to accelerate. Press SHIFT to boost."
    )

    game_description = (
        "Race your rocket through a high-speed obstacle course. Avoid collisions, "
        "manage your boost, and reach the finish line as fast as possible."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_LENGTH = 6000  # 3 stages of 2000px
        self.STAGE_LENGTH = 2000

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (15, 20, 40)
        self.COLOR_ROCKET = (255, 60, 60)
        self.COLOR_ROCKET_GLOW = (255, 150, 150)
        self.COLOR_OBSTACLE = (50, 60, 80)
        self.COLOR_OBSTACLE_OUTLINE = (100, 120, 160)
        self.COLOR_TRACK = (100, 110, 140)
        self.COLOR_BOOST_PARTICLE = (255, 180, 80)
        self.COLOR_ACCEL_PARTICLE = (150, 180, 255)
        self.COLOR_TEXT = (230, 230, 240)

        # Game constants
        self.FPS = 30
        self.TIME_LIMIT = 120  # seconds
        self.MAX_STEPS = self.TIME_LIMIT * self.FPS
        
        # Rocket physics
        self.ROCKET_SIZE = pygame.Vector2(30, 16)
        self.ROCKET_V_SPEED = 6
        self.ROCKET_ACCEL = 0.2
        self.ROCKET_FRICTION = 0.985
        self.ROCKET_MAX_SPEED = 12
        self.ROCKET_BOOST_SPEED = 25
        self.BOOST_DURATION_STEPS = int(0.5 * self.FPS)
        self.BOOST_COOLDOWN_STEPS = int(3 * self.FPS)

        # Initial state variables will be set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rocket_pos = pygame.Vector2(0, 0)
        self.rocket_vel = pygame.Vector2(0, 0)
        self.camera_x = 0
        self.obstacles = []
        self.passed_obstacles = set()
        self.particles = []
        self.stars = []
        self.is_boosting = False
        self.boost_timer = 0
        self.boost_cooldown_timer = 0
        self.obstacle_spawn_timer = 0
        self.initial_obstacle_spawn_rate = 2.0  # seconds
        self.initial_obstacle_speed = 4.0
        self.obstacle_spawn_rate = 0
        self.obstacle_speed = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.rocket_pos = pygame.Vector2(self.WIDTH * 0.2, self.HEIGHT / 2)
        self.rocket_vel = pygame.Vector2(0, 0)
        self.camera_x = 0
        
        self.obstacles = []
        self.passed_obstacles = set()
        self.particles = []

        self.is_boosting = False
        self.boost_timer = 0
        self.boost_cooldown_timer = 0

        self.obstacle_spawn_rate = self.initial_obstacle_spawn_rate
        self.obstacle_speed = self.initial_obstacle_speed
        self.obstacle_spawn_timer = self.FPS * 1.5 # Initial delay

        if not self.stars:
            for _ in range(150):
                self.stars.append({
                    'pos': pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                    'depth': self.np_random.uniform(0.1, 0.7)
                })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Vertical movement
        if movement == 1: self.rocket_pos.y -= self.ROCKET_V_SPEED
        if movement == 2: self.rocket_pos.y += self.ROCKET_V_SPEED
        self.rocket_pos.y = np.clip(self.rocket_pos.y, self.ROCKET_SIZE.y, self.HEIGHT - self.ROCKET_SIZE.y)
        
        # Horizontal acceleration
        if space_held:
            self.rocket_vel.x += self.ROCKET_ACCEL
            if self.np_random.random() < 0.5: # Spawn accel particles
                p_vel = pygame.Vector2(-self.rocket_vel.x * 1.5, self.np_random.uniform(-1, 1))
                self._spawn_particle(self.rocket_pos.x - self.ROCKET_SIZE.x / 2, self.rocket_pos.y, p_vel, 15, self.COLOR_ACCEL_PARTICLE)
        else:
            self.rocket_vel.x *= self.ROCKET_FRICTION

        # Boost
        if shift_held and self.boost_cooldown_timer <= 0 and not self.is_boosting:
            self.is_boosting = True
            self.boost_timer = self.BOOST_DURATION_STEPS
            self.boost_cooldown_timer = self.BOOST_COOLDOWN_STEPS
            reward -= 1.0 # Penalty for using boost
            # Sound: Boost activate
            for _ in range(20): # Boost activation burst
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(5, 15)
                p_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                self._spawn_particle(self.rocket_pos.x, self.rocket_pos.y, p_vel, 25, self.COLOR_BOOST_PARTICLE)


        # --- Update Game State ---
        # Update boost state
        if self.is_boosting:
            self.rocket_vel.x = self.ROCKET_BOOST_SPEED
            self.boost_timer -= 1
            if self.boost_timer <= 0:
                self.is_boosting = False
                self.rocket_vel.x = self.ROCKET_MAX_SPEED # Prevent sudden stop
            # Sound: Boosting loop
            if self.np_random.random() < 0.8: # Spawn boost trail
                p_vel = pygame.Vector2(-self.rocket_vel.x * 2, self.np_random.uniform(-2, 2))
                self._spawn_particle(self.rocket_pos.x - self.ROCKET_SIZE.x/2, self.rocket_pos.y, p_vel, 20, self.COLOR_BOOST_PARTICLE, 3)

        self.boost_cooldown_timer = max(0, self.boost_cooldown_timer - 1)
        self.rocket_vel.x = min(self.rocket_vel.x, self.ROCKET_MAX_SPEED) if not self.is_boosting else self.ROCKET_BOOST_SPEED
        
        # Update camera
        self.camera_x += self.rocket_vel.x

        # Update and spawn obstacles
        reward += self._update_obstacles()
        
        # Update particles
        self._update_particles()
        
        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_speed += 0.5
            self.obstacle_spawn_rate = max(0.4, self.obstacle_spawn_rate - 0.1)

        # --- Check for Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # Reached finish line
            time_bonus_factor = max(0, self.MAX_STEPS - self.steps) / self.MAX_STEPS
            finish_reward = 50 + 50 * time_bonus_factor # Scales from 100 down to 50
            reward += finish_reward
            self.game_over = True # Set game over to stop further steps
        elif self.game_over: # Collision happened in _update_obstacles
            reward = 0 # No reward on collision frame
            terminated = True
            
        self.score += reward
        self.steps += 1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_particle(self, x, y, vel, life, color, size=2):
        self.particles.append({
            'pos': pygame.Vector2(x, y),
            'vel': vel,
            'life': life,
            'max_life': life,
            'color': color,
            'size': size
        })
        
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_obstacles(self):
        reward = 0
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self._spawn_obstacle()
            self.obstacle_spawn_timer = int(self.obstacle_spawn_rate * self.FPS)

        rocket_world_x = self.camera_x + self.rocket_pos.x
        rocket_rect = pygame.Rect(
            self.rocket_pos.x - self.ROCKET_SIZE.x / 2,
            self.rocket_pos.y - self.ROCKET_SIZE.y / 2,
            self.ROCKET_SIZE.x, self.ROCKET_SIZE.y
        )

        for obs in self.obstacles[:]:
            obs['pos'].x -= obs['vel_x']
            
            # Check for passing
            if obs['id'] not in self.passed_obstacles and obs['pos'].x + obs['size'].x < rocket_world_x:
                reward += 5.0
                self.passed_obstacles.add(obs['id'])
                # Sound: Point score
            
            # Check for collision
            obs_screen_rect = pygame.Rect(
                obs['pos'].x - self.camera_x, obs['pos'].y,
                obs['size'].x, obs['size'].y
            )
            if rocket_rect.colliderect(obs_screen_rect):
                self.game_over = True
                # Sound: Explosion
                for _ in range(50):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(2, 12)
                    p_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                    self._spawn_particle(self.rocket_pos.x, self.rocket_pos.y, p_vel, 30, self.COLOR_ROCKET_GLOW, self.np_random.integers(1, 4))

            # Remove if off-screen
            if obs['pos'].x - self.camera_x + obs['size'].x < 0:
                self.obstacles.remove(obs)
        return reward

    def _spawn_obstacle(self):
        track_height = self.HEIGHT - 80
        height = self.np_random.uniform(track_height * 0.15, track_height * 0.4)
        y_pos = self.np_random.uniform(40, self.HEIGHT - 40 - height)
        width = self.np_random.uniform(30, 80)
        
        obstacle = {
            'id': self.steps + self.np_random.random(),
            'pos': pygame.Vector2(self.camera_x + self.WIDTH + 50, y_pos),
            'size': pygame.Vector2(width, height),
            'vel_x': self.obstacle_speed if self.np_random.random() > 0.7 else 0 # 30% are static
        }
        self.obstacles.append(obstacle)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.camera_x + self.rocket_pos.x >= self.WORLD_LENGTH:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Parallax stars
        for star in self.stars:
            star_x = (star['pos'].x - self.camera_x * star['depth']) % self.WIDTH
            pygame.gfxdraw.pixel(self.screen, int(star_x), int(star['pos'].y), (255,255,255, int(150 * star['depth'])))

        # Track boundaries
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, 30), (self.WIDTH, 30), 2)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.HEIGHT - 30), (self.WIDTH, self.HEIGHT - 30), 2)

        # Finish line
        finish_screen_x = self.WORLD_LENGTH - self.camera_x
        if finish_screen_x < self.WIDTH:
            check_size = 20
            for y in range(0, self.HEIGHT, check_size):
                color = (255, 255, 255) if (y // check_size) % 2 == 0 else (0, 0, 0)
                pygame.draw.rect(self.screen, color, (finish_screen_x, y, check_size, check_size))

        # Obstacles
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['pos'].x - self.camera_x, obs['pos'].y, obs['size'].x, obs['size'].y)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, obs_rect, 2, border_radius=3)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size'] * (p['life'] / p['max_life'])), color)

        # Rocket
        if not (self.game_over and self._check_termination()): # Don't draw if crashed
            rocket_points = [
                (self.rocket_pos.x + self.ROCKET_SIZE.x / 2, self.rocket_pos.y),
                (self.rocket_pos.x - self.ROCKET_SIZE.x / 2, self.rocket_pos.y - self.ROCKET_SIZE.y / 2),
                (self.rocket_pos.x - self.ROCKET_SIZE.x / 2, self.rocket_pos.y + self.ROCKET_SIZE.y / 2),
            ]
            # Glow effect
            pygame.gfxdraw.aapolygon(self.screen, rocket_points, self.COLOR_ROCKET_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, rocket_points, self.COLOR_ROCKET_GLOW)
            # Main body
            pygame.gfxdraw.aapolygon(self.screen, rocket_points, self.COLOR_ROCKET)
            pygame.gfxdraw.filled_polygon(self.screen, rocket_points, self.COLOR_ROCKET)
    
    def _render_ui(self):
        # Time
        time_left = max(0, self.TIME_LIMIT - (self.steps / self.FPS))
        time_text = f"TIME: {time_left:.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 5))

        # Progress/Stage
        current_stage = int(self.camera_x // self.STAGE_LENGTH) + 1
        stage_text = f"STAGE: {current_stage}/3"
        stage_surf = self.font_ui.render(stage_text, True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, (self.WIDTH - stage_surf.get_width() - 10, 5))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH / 2 - score_surf.get_width() / 2, 5))

        # Boost Meter
        boost_meter_width = 150
        boost_meter_height = 10
        boost_meter_y = self.HEIGHT - 20
        # Background
        pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (10, boost_meter_y, boost_meter_width, boost_meter_height), border_radius=3)
        # Fill
        fill_width = boost_meter_width * (1 - self.boost_cooldown_timer / self.BOOST_COOLDOWN_STEPS)
        if self.boost_cooldown_timer > 0:
            fill_color = (100, 100, 150) # Cooldown color
        else:
            fill_color = self.COLOR_BOOST_PARTICLE # Ready color
        pygame.draw.rect(self.screen, fill_color, (10, boost_meter_y, fill_width, boost_meter_height), border_radius=3)

        # Game Over / Win Message
        if self.game_over and self._check_termination():
            if self.camera_x + self.rocket_pos.x >= self.WORLD_LENGTH:
                msg = "FINISH!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = self.COLOR_ROCKET
            
            msg_surf = self.font_big.render(msg, True, color)
            self.screen.blit(msg_surf, (self.WIDTH / 2 - msg_surf.get_width() / 2, self.HEIGHT / 2 - msg_surf.get_height() / 2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": int(self.camera_x // self.STAGE_LENGTH) + 1,
            "world_progress": (self.camera_x + self.rocket_pos.x) / self.WORLD_LENGTH
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Rocket Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(env.FPS)
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            
    env.close()