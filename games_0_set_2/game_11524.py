import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:07:08.124454
# Source Brief: brief_01524.md
# Brief Index: 1524
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# A helper class for particles to keep the main loop clean
class Particle:
    def __init__(self, x, y, color, min_speed, max_speed, lifespan):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(min_speed, max_speed)
        self.pos = [x, y]
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.lifespan = lifespan
        self.initial_lifespan = lifespan
        self.color = color
        self.radius = random.uniform(2, 5)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.98 # friction
        self.vel[1] *= 0.98
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            # Fade out effect
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            alpha = max(0, min(255, alpha))
            
            # Create a temporary surface for transparency
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)))

class GameEnv(gym.Env):
    game_description = (
        "Guide your player down a tilting track, collecting stars and hitting boosters "
        "while trying not to fall off the edge."
    )
    user_guide = "Controls: Use the ← and → arrow keys to tilt the track left and right."
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FINISH_Y = 380
    MAX_STEPS = 1000
    WIN_SCORE = 100

    # Colors
    COLOR_BG = (10, 10, 10) # Black
    COLOR_TRACK = (40, 40, 40) # Dark Gray
    COLOR_PLAYER = (0, 170, 255) # Bright Blue
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_STAR = (255, 255, 0) # Yellow
    COLOR_BOOSTER = (255, 68, 68) # Red
    COLOR_TEXT = (255, 255, 255) # White
    COLOR_FINISH_LINE = (200, 200, 200)

    # Game Physics
    TILT_CHANGE_RATE = 2.0  # Degrees per step
    MAX_TILT_ANGLE = 70.0
    TILT_FORCE_MULTIPLIER = 0.2
    PLAYER_FRICTION = 0.95
    INITIAL_DOWNWARD_SPEED = 1.0
    SPEED_INCREASE_PER_100_STEPS = 0.05
    BOOSTER_SPEED_MULTIPLIER = 2.5
    BOOSTER_DURATION = 90 # steps (3 seconds at 30fps)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("Consolas", 30)
            self.font_small = pygame.font.SysFont("Consolas", 20)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 28)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.track_tilt_angle = 0.0
        self.player_pos = [0.0, 0.0]
        self.player_vel = [0.0, 0.0]
        self.base_downward_speed = 0.0
        self.booster_timer = 0
        self.booster_spawn_rate = 0
        self.stars = []
        self.boosters = []
        self.particles = []
        
        self.np_random = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.track_tilt_angle = 0.0
        
        self.player_pos = [self.SCREEN_WIDTH / 2, 50.0]
        self.player_vel = [0.0, self.INITIAL_DOWNWARD_SPEED]
        self.base_downward_speed = self.INITIAL_DOWNWARD_SPEED
        
        self.booster_timer = 0
        self.booster_spawn_rate = 1

        self.particles = []
        self._generate_collectibles()
        
        return self._get_observation(), self._get_info()

    def _generate_collectibles(self):
        self.stars = []
        self.boosters = []
        
        track_center_x = self.SCREEN_WIDTH / 2
        track_width = 100

        # Generate Stars
        for _ in range(15):
            y = random.uniform(100, self.FINISH_Y - 20)
            x_offset = random.uniform(-track_width, track_width)
            self.stars.append([track_center_x + x_offset, y])

        # Generate Boosters
        for _ in range(5):
            y = random.uniform(120, self.FINISH_Y - 40)
            x_offset = random.uniform(-track_width, track_width)
            self.boosters.append([track_center_x + x_offset, y])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack Action & Update Game State ---
        movement = action[0]
        
        if movement == 3: # Left
            self.track_tilt_angle -= self.TILT_CHANGE_RATE
        elif movement == 4: # Right
            self.track_tilt_angle += self.TILT_CHANGE_RATE
        
        self.track_tilt_angle = np.clip(self.track_tilt_angle, -self.MAX_TILT_ANGLE, self.MAX_TILT_ANGLE)

        # --- 2. Update Physics ---
        self._update_player_physics()
        self._update_particles()
        
        # --- 3. Handle Collisions and Events ---
        reward = self._handle_collisions()
        
        # --- 4. Calculate Rewards & Check Termination ---
        self.steps += 1
        reward += 0.01 # Small reward for surviving
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        # Terminal rewards
        if terminated:
            if abs(self.track_tilt_angle) >= self.MAX_TILT_ANGLE:
                reward = -50.0 # Fell off
            elif self.player_pos[1] >= self.FINISH_Y and self.score >= self.WIN_SCORE:
                reward = 100.0 # Won
            elif self.player_pos[1] >= self.FINISH_Y:
                reward = 10.0 # Finished but didn't win
        
        if terminated or truncated:
            self.game_over = True
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player_physics(self):
        # Update downward speed (difficulty scaling)
        if self.steps > 0 and self.steps % 100 == 0:
            self.base_downward_speed += self.SPEED_INCREASE_PER_100_STEPS
        
        # Handle booster effect
        if self.booster_timer > 0:
            self.booster_timer -= 1
            downward_speed = self.base_downward_speed * self.BOOSTER_SPEED_MULTIPLIER
        else:
            downward_speed = self.base_downward_speed
        
        # Calculate horizontal force from tilt
        horizontal_force = math.sin(math.radians(self.track_tilt_angle)) * self.TILT_FORCE_MULTIPLIER
        
        # Update velocity
        self.player_vel[0] += horizontal_force
        self.player_vel[0] *= self.PLAYER_FRICTION # Apply friction
        self.player_vel[1] = downward_speed
        
        # Update position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        
        # Clamp player to track bounds
        track_width = 150 # Visual track width
        track_center_x = self.SCREEN_WIDTH / 2
        self.player_pos[0] = np.clip(self.player_pos[0], track_center_x - track_width, track_center_x + track_width)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - 8, self.player_pos[1] - 8, 16, 16)

        # Star collisions
        for star in self.stars[:]:
            star_rect = pygame.Rect(star[0] - 6, star[1] - 6, 12, 12)
            if player_rect.colliderect(star_rect):
                self.stars.remove(star)
                self.score += 5
                reward += 5.0
                self._create_particles(star[0], star[1], self.COLOR_STAR, 15)

        # Booster collisions
        for booster in self.boosters[:]:
            booster_rect = pygame.Rect(booster[0] - 7, booster[1] - 7, 14, 14)
            if player_rect.colliderect(booster_rect):
                self.boosters.remove(booster)
                self.booster_timer = self.BOOSTER_DURATION
                reward += 1.0
                self._create_particles(booster[0], booster[1], self.COLOR_BOOSTER, 25)

        return reward

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, 1, 4, 40))

    def _check_termination(self):
        if abs(self.track_tilt_angle) >= self.MAX_TILT_ANGLE:
            return True
        if self.player_pos[1] >= self.FINISH_Y:
            return True
        if self.score >= self.WIN_SCORE and self.player_pos[1] >= self.FINISH_Y:
             return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tilt_angle": self.track_tilt_angle,
            "player_y": self.player_pos[1],
        }

    def _render_game(self):
        # Draw track
        track_center_x = self.SCREEN_WIDTH / 2
        track_width = 160
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (track_center_x - track_width, 0, track_width * 2, self.SCREEN_HEIGHT))

        # Draw finish line
        pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (track_center_x - track_width, self.FINISH_Y), (track_center_x + track_width, self.FINISH_Y), 3)

        # Draw collectibles
        for star_pos in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (int(star_pos[0]), int(star_pos[1])), 6)
        
        for booster_pos in self.boosters:
            x, y = int(booster_pos[0]), int(booster_pos[1])
            points = [(x, y - 8), (x - 7, y + 4), (x + 7, y + 4)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BOOSTER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BOOSTER)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        player_size = 8
        
        # Glow effect
        glow_radius = player_size * 2.5 if self.booster_timer > 0 else player_size * 1.8
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = self.COLOR_BOOSTER if self.booster_timer > 0 else self.COLOR_PLAYER_GLOW
        pygame.draw.circle(glow_surf, (*glow_color, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (px - glow_radius, py - glow_radius))

        # Player core
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px - player_size, py - player_size, player_size * 2, player_size * 2))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Tilt angle bar
        bar_x, bar_y, bar_w, bar_h = self.SCREEN_WIDTH / 2 - 100, 20, 200, 20
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (bar_x, bar_y, bar_w, bar_h))
        
        tilt_ratio = (self.track_tilt_angle + self.MAX_TILT_ANGLE) / (2 * self.MAX_TILT_ANGLE)
        indicator_x = bar_x + tilt_ratio * bar_w
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (indicator_x, bar_y), (indicator_x, bar_y + bar_h), 3)
        pygame.draw.line(self.screen, self.COLOR_BOOSTER, (bar_x, bar_y + bar_h/2), (bar_x - 10, bar_y + bar_h/2), 3)
        pygame.draw.line(self.screen, self.COLOR_BOOSTER, (bar_x + bar_w, bar_y + bar_h/2), (bar_x + bar_w + 10, bar_y + bar_h/2), 3)

        # Tilt angle text
        tilt_text = self.font_small.render(f"{self.track_tilt_angle:.1f}°", True, self.COLOR_TEXT)
        text_rect = tilt_text.get_rect(center=(self.SCREEN_WIDTH / 2, bar_y + bar_h + 15))
        self.screen.blit(tilt_text, text_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = ""
            if abs(self.track_tilt_angle) >= self.MAX_TILT_ANGLE:
                message = "FELL OFF!"
            elif self.player_pos[1] >= self.FINISH_Y and self.score >= self.WIN_SCORE:
                message = "YOU WIN!"
            elif self.steps >= self.MAX_STEPS:
                message = "TIME UP!"
            else:
                message = "FINISH!"

            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tilt Fall")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # A/Left Arrow -> Tilt Left (action 3)
    # D/Right Arrow -> Tilt Right (action 4)
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action[0] = 4
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
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

        clock.tick(30) # Run at 30 FPS
        
    env.close()