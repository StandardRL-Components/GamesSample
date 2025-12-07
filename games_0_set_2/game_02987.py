
# Generated: 2025-08-28T06:39:25.281641
# Source Brief: brief_02987.md
# Brief Index: 2987

        
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
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to steer. "
        "Hold Space for a speed boost. Hold Shift to drift through corners."
    )

    game_description = (
        "A fast-paced, retro-futuristic arcade racer. Navigate a procedurally generated neon track, "
        "dodge obstacles, and aim for the fastest time to the finish line."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (10, 0, 20)
    COLOR_STAR = (200, 200, 255)
    COLOR_TRACK_BORDER = (0, 120, 255)
    COLOR_TRACK_FILL = (0, 20, 80)
    COLOR_CAR = (255, 0, 128)
    COLOR_CAR_GLOW = (255, 100, 180)
    COLOR_OBSTACLE = (255, 165, 0)
    COLOR_OBSTACLE_GLOW = (255, 165, 0, 100)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_BOOST_BAR = (0, 255, 255)
    COLOR_BOOST_BAR_BG = (50, 50, 50)
    
    # Game parameters
    FPS = 30
    MAX_STEPS = 3000
    TRACK_LENGTH_PIXELS = 15000
    CHECKPOINT_INTERVAL = 2500

    # Car physics
    ACCEL_RATE = 0.4
    BRAKE_RATE = 0.8
    MAX_SPEED = 12.0
    FRICTION = 0.98  # Multiplier for speed each frame
    STEER_ANGLE_RATE = 0.05
    MAX_STEER_ANGLE = 0.8
    DRIFT_SLIP = 0.8
    DRIFT_STEER_MULTIPLIER = 1.5

    # Boost
    BOOST_ACCEL = 1.0
    MAX_BOOST_FUEL = 100
    BOOST_CONSUMPTION = 2.0
    BOOST_RECHARGE = 0.4
    
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.car_pos = None
        self.car_speed = None
        self.car_angle = None
        self.car_steer_angle = None
        self.camera_x = None
        self.track = None
        self.obstacles = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.boost_fuel = None
        self.next_checkpoint = None
        self.obstacle_density = None
        self.track_complexity = None
        
        self.np_random = None
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.car_pos = pygame.Vector2(100, self.SCREEN_HEIGHT / 2)
        self.car_speed = 0.0
        self.car_angle = 0.0
        self.car_steer_angle = 0.0
        self.camera_x = 0
        
        self.boost_fuel = self.MAX_BOOST_FUEL
        self.next_checkpoint = self.CHECKPOINT_INTERVAL
        self.obstacle_density = 0.05
        self.track_complexity = 1.0

        self._generate_track()
        self.obstacles = []
        self.particles = []
        self.stars = [
            (random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT), random.uniform(0.1, 0.5))
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        
        self._handle_input(movement, space_held, shift_held)
        self._update_physics(shift_held)
        self._update_world()
        
        reward += 0.1  # Survival reward
        if self.car_speed < self.MAX_SPEED * 0.2:
            reward -= 0.2

        terminated = self._check_collisions()
        if terminated:
            reward = -100.0
            self.game_over = True
        
        if not terminated:
            if self.car_pos.x > self.next_checkpoint:
                reward += 5.0
                self.score += 5
                self.next_checkpoint += self.CHECKPOINT_INTERVAL
            
            if self.car_pos.x >= self.TRACK_LENGTH_PIXELS:
                terminated = True
                self.game_won = True
                time_bonus = (self.MAX_STEPS - self.steps) / self.MAX_STEPS
                reward = 100.0 + 50.0 * time_bonus
                self.score += int(reward)

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Acceleration
        if movement == 1:  # Up
            self.car_speed += self.ACCEL_RATE
        elif movement == 2:  # Down
            self.car_speed -= self.BRAKE_RATE
            # sfx: brake screech
        
        # Steering
        steer_input = 0
        if movement == 3:  # Left
            steer_input = -1
        elif movement == 4:  # Right
            steer_input = 1

        steer_multiplier = self.DRIFT_STEER_MULTIPLIER if shift_held and self.car_speed > self.MAX_SPEED * 0.3 else 1.0
        self.car_steer_angle = steer_input * self.MAX_STEER_ANGLE * steer_multiplier

        # Boost
        if space_held and self.boost_fuel > 0:
            self.car_speed += self.BOOST_ACCEL
            self.boost_fuel = max(0, self.boost_fuel - self.BOOST_CONSUMPTION)
            # sfx: boost whoosh
            self._spawn_particles(self.car_pos, 5, (0, 255, 255), 2.0, -self.car_angle)
        else:
            self.boost_fuel = min(self.MAX_BOOST_FUEL, self.boost_fuel + self.BOOST_RECHARGE)

    def _update_physics(self, is_drifting):
        # Apply friction and clamp speed
        self.car_speed *= self.FRICTION
        self.car_speed = max(0, min(self.car_speed, self.MAX_SPEED))
        
        if self.car_speed > 0.1:
            # Update car angle based on steering
            turn_rate = self.car_steer_angle * (self.car_speed / self.MAX_SPEED)
            self.car_angle += turn_rate
            
            # Update position
            velocity = pygame.Vector2(math.cos(self.car_angle), math.sin(self.car_angle)) * self.car_speed
            
            if is_drifting and abs(self.car_steer_angle) > 0.1 and self.car_speed > self.MAX_SPEED * 0.3:
                # sfx: drift tire screech
                # Apply sideways slip
                right_vector = pygame.Vector2(math.cos(self.car_angle + math.pi/2), math.sin(self.car_angle + math.pi/2))
                slip_amount = self.car_steer_angle * self.DRIFT_SLIP * (self.car_speed / self.MAX_SPEED)
                velocity += right_vector * slip_amount
                self.car_pos += velocity
                
                # Spawn drift particles
                particle_pos = self.car_pos - pygame.Vector2(math.cos(self.car_angle), math.sin(self.car_angle)) * 10
                self._spawn_particles(particle_pos, 2, (200,200,200), 1.0, self.car_angle + math.pi)
            else:
                self.car_pos += velocity
        
        # Dampen car angle and steering angle
        self.car_angle *= 0.9
        self.car_steer_angle *= 0.8

    def _update_world(self):
        # Camera follows car
        self.camera_x = self.car_pos.x - 150

        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_density = min(0.2, self.obstacle_density + 0.02)
            self.track_complexity = min(2.0, self.track_complexity + 0.1)
            self._generate_track() # Regenerate track with new complexity
        
        # Generate new obstacles
        last_obs_x = self.obstacles[-1][0].x if self.obstacles else 0
        if self.car_pos.x + self.SCREEN_WIDTH > last_obs_x:
            self._generate_obstacles(self.car_pos.x + self.SCREEN_WIDTH, 2000)

        # Remove off-screen obstacles and particles
        self.obstacles = [obs for obs in self.obstacles if obs[0].x > self.camera_x - 50]
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_collisions(self):
        car_rect = pygame.Rect(self.car_pos.x - 8, self.car_pos.y - 4, 16, 8)
        
        # Obstacle collision
        for obs_rect, _ in self.obstacles:
            if car_rect.colliderect(obs_rect):
                # sfx: explosion
                self._spawn_particles(self.car_pos, 50, self.COLOR_OBSTACLE, 4.0)
                return True
        
        # Track bounds collision
        track_y, track_width = self._get_track_props_at(self.car_pos.x)
        half_width = track_width / 2
        if not (track_y - half_width < self.car_pos.y < track_y + half_width):
            # sfx: crash
            self._spawn_particles(self.car_pos, 30, self.COLOR_TRACK_BORDER, 3.0)
            return True
            
        return False
        
    def _generate_track(self):
        self.track = []
        y = self.SCREEN_HEIGHT / 2
        base_width = 120
        amp1 = self.np_random.uniform(50, 100) * self.track_complexity
        freq1 = self.np_random.uniform(0.0005, 0.001)
        phase1 = self.np_random.uniform(0, math.pi * 2)
        amp2 = self.np_random.uniform(20, 40) * self.track_complexity
        freq2 = self.np_random.uniform(0.002, 0.004)
        phase2 = self.np_random.uniform(0, math.pi * 2)

        for x in range(int(self.TRACK_LENGTH_PIXELS) + self.SCREEN_WIDTH):
            y_offset = amp1 * math.sin(freq1 * x + phase1) + amp2 * math.sin(freq2 * x + phase2)
            center_y = self.SCREEN_HEIGHT / 2 + y_offset
            width = base_width - abs(y_offset) * 0.1
            self.track.append((center_y, width))

    def _generate_obstacles(self, start_x, length):
        x = start_x
        while x < start_x + length:
            x += self.np_random.uniform(100, 400) / (self.obstacle_density * 20 + 1)
            if self.np_random.random() < self.obstacle_density:
                track_y, track_width = self._get_track_props_at(x)
                obs_width = self.np_random.uniform(20, 40)
                obs_height = self.np_random.uniform(20, 40)
                
                # Place obstacle on either side of the track center
                side = self.np_random.choice([-1, 1])
                offset = self.np_random.uniform(0, track_width / 2 - obs_width / 2)
                obs_y = track_y + side * offset - obs_height / 2
                
                rect = pygame.Rect(x, obs_y, obs_width, obs_height)
                self.obstacles.append((rect, self.COLOR_OBSTACLE))

    def _get_track_props_at(self, x):
        idx = int(max(0, min(len(self.track) - 1, x)))
        return self.track[idx]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render stars with parallax
        for x, y, speed in self.stars:
            screen_x = (x - self.camera_x * speed) % self.SCREEN_WIDTH
            pygame.gfxdraw.pixel(self.screen, int(screen_x), int(y), self.COLOR_STAR)

        # Render track
        start_idx = max(0, int(self.camera_x))
        end_idx = min(len(self.track), start_idx + self.SCREEN_WIDTH + 2)
        for x_offset in range(end_idx - start_idx - 1):
            x = start_idx + x_offset
            screen_x = int(x - self.camera_x)
            
            y1, w1 = self.track[x]
            y2, w2 = self.track[x+1]
            
            # Track fill
            pygame.draw.polygon(self.screen, self.COLOR_TRACK_FILL, [
                (screen_x, y1 - w1/2), (screen_x + 1, y2 - w2/2),
                (screen_x + 1, y2 + w2/2), (screen_x, y1 + w1/2)
            ])
            
            # Track borders (anti-aliased)
            pygame.gfxdraw.line(self.screen, screen_x, int(y1 - w1/2), screen_x + 1, int(y2 - w2/2), self.COLOR_TRACK_BORDER)
            pygame.gfxdraw.line(self.screen, screen_x, int(y1 + w1/2), screen_x + 1, int(y2 + w2/2), self.COLOR_TRACK_BORDER)

        # Render obstacles
        for obs_rect, color in self.obstacles:
            screen_rect = obs_rect.move(-self.camera_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.SCREEN_WIDTH:
                # Glow effect
                glow_rect = screen_rect.inflate(10, 10)
                glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, self.COLOR_OBSTACLE_GLOW, (0, 0, *glow_rect.size), border_radius=5)
                self.screen.blit(glow_surf, glow_rect.topleft)
                # Obstacle
                pygame.draw.rect(self.screen, color, screen_rect, border_radius=3)

        # Render particles
        for p in self.particles:
            screen_pos = p['pos'] - pygame.Vector2(self.camera_x, 0)
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, screen_pos - pygame.Vector2(p['size'], p['size']))

        # Render car
        car_screen_pos = self.car_pos - pygame.Vector2(self.camera_x, 0)
        car_poly = [
            pygame.Vector2(10, 0), pygame.Vector2(-8, 5),
            pygame.Vector2(-5, 0), pygame.Vector2(-8, -5)
        ]
        
        # Tilt car based on steering
        tilt_angle = self.car_steer_angle * 0.5
        
        rotated_poly = [p.rotate_rad(self.car_angle + tilt_angle) + car_screen_pos for p in car_poly]
        
        # Car glow
        pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in rotated_poly], self.COLOR_CAR_GLOW)
        # Car body
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in rotated_poly], self.COLOR_CAR)


    def _render_ui(self):
        # Time/Steps
        time_text = f"TIME: {self.steps / self.FPS:.2f}s"
        text_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 40))
        
        # Boost bar
        bar_width = 150
        bar_height = 15
        boost_fill = (self.boost_fuel / self.MAX_BOOST_FUEL) * bar_width
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR_BG, (self.SCREEN_WIDTH - bar_width - 10, 10, bar_width, bar_height))
        if boost_fill > 0:
            pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR, (self.SCREEN_WIDTH - bar_width - 10, 10, boost_fill, bar_height))

        # Progress bar
        progress = self.car_pos.x / self.TRACK_LENGTH_PIXELS
        prog_bar_width = self.SCREEN_WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR_BG, (10, self.SCREEN_HEIGHT - 20, prog_bar_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_TRACK_BORDER, (10, self.SCREEN_HEIGHT - 20, prog_bar_width * progress, 10))

        if self.game_over:
            msg = "CRASHED"
            color = self.COLOR_OBSTACLE
            if self.game_won:
                msg = "FINISH!"
                color = self.COLOR_BOOST_BAR
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "car_speed": self.car_speed,
            "boost_fuel": self.boost_fuel,
            "car_x_pos": self.car_pos.x
        }

    def _spawn_particles(self, pos, count, color, speed_mult, angle_offset=None):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi) if angle_offset is None else angle_offset + random.uniform(-0.5, 0.5)
            speed = random.uniform(1.0, 3.0) * speed_mult
            life = random.randint(10, 25)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.randint(1, 4)
            })

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Main game loop for human play
    while not terminated:
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
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
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    pygame.time.wait(2000)
    env.close()