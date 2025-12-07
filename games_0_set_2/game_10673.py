import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:47:53.177699
# Source Brief: brief_00673.md
# Brief Index: 673
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a visually striking arcade game.
    The player controls a pulsating fractal, rotating it and reinforcing
    its segments to avoid colliding with static obstacles. The game's
    difficulty increases over time as the fractal pulsates faster and
    more obstacles appear.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a pulsating fractal, rotating it to avoid colliding with static obstacles. "
        "Reinforce segments to survive as the game speeds up and more obstacles appear."
    )
    user_guide = (
        "Controls: Use ↑ and ↓ arrow keys to rotate the fractal. "
        "Press space to reinforce a random segment, making it stronger."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 10000
    VICTORY_TIME_SECONDS = 60

    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_OBSTACLE = (70, 70, 80)
    COLOR_OBSTACLE_BORDER = (90, 90, 100)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TIMER = (255, 255, 100)

    # Fractal settings
    FRACTAL_SEGMENTS = 8
    FRACTAL_ROTATION_SPEED = 2.0  # degrees per step
    FRACTAL_MAX_RADIUS = 150
    FRACTAL_BASE_RADIUS = 20

    # Game progression
    INITIAL_OSCILLATION_HZ = 0.5
    HZ_INCREASE_PER_LEVEL = 0.05
    LEVEL_UP_INTERVAL_STEPS = 10 * FPS  # Every 10 seconds
    OBSTACLE_SPAWN_INTERVAL_STEPS = 20 * FPS # Every 20 seconds
    INITIAL_OBSTACLES = 10

    # Reinforcement
    REINFORCE_COOLDOWN_STEPS = 15

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0.0
        self.level = 1
        self.oscillation_speed_hz = self.INITIAL_OSCILLATION_HZ
        self.fractal_angle = 0.0
        self.segment_health = []
        self.obstacles = []
        self.particles = []
        self.reinforce_cooldown = 0
        self.next_level_step = 0
        self.next_obstacle_spawn_step = 0

        # self.reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0.0
        self.level = 1
        self.oscillation_speed_hz = self.INITIAL_OSCILLATION_HZ
        self.fractal_angle = self.np_random.uniform(0, 360)
        self.segment_health = [1] * self.FRACTAL_SEGMENTS
        self.particles = []
        self.reinforce_cooldown = 0

        self.next_level_step = self.LEVEL_UP_INTERVAL_STEPS
        self.next_obstacle_spawn_step = self.OBSTACLE_SPAWN_INTERVAL_STEPS

        self._generate_obstacles(self.INITIAL_OBSTACLES)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        terminated = False
        truncated = False

        # --- Action Handling ---
        movement, space_held, _ = action
        # 1=up(CCW), 2=down(CW)
        if movement == 1:
            self.fractal_angle -= self.FRACTAL_ROTATION_SPEED
        elif movement == 2:
            self.fractal_angle += self.FRACTAL_ROTATION_SPEED

        if space_held == 1 and self.reinforce_cooldown <= 0:
            if self._reinforce_segment():
                reward += 1.0
                self.reinforce_cooldown = self.REINFORCE_COOLDOWN_STEPS
                # SFX: Reinforce sound

        # --- Game Logic Update ---
        self.steps += 1
        self.timer += 1.0 / self.FPS
        if self.reinforce_cooldown > 0:
            self.reinforce_cooldown -= 1

        self._update_particles()
        self._update_difficulty()

        # --- Collision Detection & State Update ---
        collision, reward_mod = self._check_collisions()
        reward += reward_mod

        if collision:
            terminated = True
            self.game_over = True
            reward = -100.0
            # SFX: Explosion/Failure sound

        # --- Survival & Win Condition ---
        if not terminated:
            reward += 0.1  # Survival reward
            if self.timer >= self.VICTORY_TIME_SECONDS:
                terminated = True
                self.game_over = True
                reward += 100.0
                # SFX: Victory fanfare
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "timer": self.timer,
            "level": self.level,
        }

    def _generate_obstacles(self, num_obstacles):
        self.obstacles = []
        min_dist_from_center = self.FRACTAL_MAX_RADIUS + 30
        for _ in range(num_obstacles):
            while True:
                w = self.np_random.integers(30, 80)
                h = self.np_random.integers(30, 80)
                x = self.np_random.integers(0, self.SCREEN_WIDTH - w)
                y = self.np_random.integers(0, self.SCREEN_HEIGHT - h)
                rect = pygame.Rect(x, y, w, h)
                
                center_x, center_y = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2
                dist_sq = (rect.centerx - center_x)**2 + (rect.centery - center_y)**2
                
                if dist_sq > min_dist_from_center**2 and not any(rect.colliderect(obs) for obs in self.obstacles):
                    self.obstacles.append(rect)
                    break

    def _add_obstacle(self):
        self._generate_obstacles(len(self.obstacles) + 1)
        # SFX: Obstacle spawn sound

    def _update_difficulty(self):
        if self.steps >= self.next_level_step:
            self.level += 1
            self.oscillation_speed_hz += self.HZ_INCREASE_PER_LEVEL
            self.next_level_step += self.LEVEL_UP_INTERVAL_STEPS
            # SFX: Level up chime

        if self.steps >= self.next_obstacle_spawn_step:
            self._add_obstacle()
            self.next_obstacle_spawn_step += self.OBSTACLE_SPAWN_INTERVAL_STEPS

    def _reinforce_segment(self):
        available_segments = [i for i, h in enumerate(self.segment_health) if h == 1]
        if not available_segments:
            return False
        
        segment_to_reinforce = self.np_random.choice(available_segments)
        self.segment_health[segment_to_reinforce] = 2
        
        # Add visual feedback particles
        angle_rad = math.radians(self.fractal_angle + (segment_to_reinforce * 360 / self.FRACTAL_SEGMENTS))
        for _ in range(15):
            self._create_particle(
                pos=pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2),
                color=(255, 255, 150),
                speed_mult=self.np_random.uniform(2, 5),
                lifespan=15,
                direction=angle_rad,
                spread=0.4
            )
        return True

    def _check_collisions(self):
        center = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        time_factor = self.steps * self.oscillation_speed_hz * 2 * math.pi / self.FPS
        pulse = (math.sin(time_factor) + 1) / 2  # Normalized to 0-1
        current_radius = self.FRACTAL_BASE_RADIUS + pulse * (self.FRACTAL_MAX_RADIUS - self.FRACTAL_BASE_RADIUS)

        for i in range(self.FRACTAL_SEGMENTS):
            if self.segment_health[i] <= 0: continue

            angle_deg = self.fractal_angle + (i * 360 / self.FRACTAL_SEGMENTS)
            angle_rad = math.radians(angle_deg)
            
            end_point = center + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * current_radius
            
            for obs_rect in self.obstacles:
                if obs_rect.clipline(center, end_point):
                    self.segment_health[i] -= 1
                    # SFX: Segment hit sound
                    
                    # Create collision particles
                    clip_points = obs_rect.clipline(center, end_point)
                    if clip_points:
                        collision_point = pygame.Vector2(clip_points[0])
                        for _ in range(20):
                            self._create_particle(
                                pos=collision_point,
                                color=(255, 100, 80),
                                speed_mult=self.np_random.uniform(1, 4),
                                lifespan=25,
                                spread=math.pi
                            )
                    
                    if self.segment_health[i] <= 0:
                        # Check if all segments are destroyed
                        if all(h <= 0 for h in self.segment_health):
                            return True, 0 # Collision is fatal
        return False, 0

    def _create_particle(self, pos, color, speed_mult, lifespan, direction=None, spread=2*math.pi):
        if direction is None:
            angle = self.np_random.uniform(0, spread)
        else:
            angle = self.np_random.normal(direction, spread / 2)
        
        speed = self.np_random.uniform(1, 2) * speed_mult
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.particles.append({
            "pos": pos,
            "vel": vel,
            "lifespan": lifespan,
            "max_lifespan": lifespan,
            "color": color,
            "radius": self.np_random.uniform(2, 5)
        })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95  # friction
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _render_game(self):
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_BORDER, obs, 2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (*p["color"], alpha)
            radius = p["radius"] * (p["lifespan"] / p["max_lifespan"])
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(radius), color
            )

        # Draw fractal
        center = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        time_factor = self.steps * self.oscillation_speed_hz * 2 * math.pi / self.FPS
        pulse = (math.sin(time_factor) + 1) / 2
        current_radius = self.FRACTAL_BASE_RADIUS + pulse * (self.FRACTAL_MAX_RADIUS - self.FRACTAL_BASE_RADIUS)
        
        for i in range(self.FRACTAL_SEGMENTS):
            if self.segment_health[i] <= 0:
                continue

            angle_deg = self.fractal_angle + (i * 360 / self.FRACTAL_SEGMENTS)
            angle_rad = math.radians(angle_deg)
            end_point = center + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * current_radius
            
            # Color based on segment index
            hue = int((i / self.FRACTAL_SEGMENTS) * 360)
            base_color = pygame.Color(0)
            base_color.hsla = (hue, 100, 50, 100)

            # Reinforced segments are brighter
            if self.segment_health[i] == 2:
                final_color = pygame.Color(0)
                final_color.hsla = (hue, 100, 75, 100)
                width = 4
            else:
                final_color = base_color
                width = 2
            
            # Glow effect
            for j in range(4, 0, -1):
                glow_color = (*final_color[:3], 15 * j)
                pygame.draw.line(self.screen, glow_color, center, end_point, width + j * 3)

            # Main line
            pygame.draw.line(self.screen, final_color, center, end_point, width)
            pygame.gfxdraw.filled_circle(self.screen, int(end_point.x), int(end_point.y), 4, final_color)

        pygame.gfxdraw.filled_circle(self.screen, int(center.x), int(center.y), 8, (255, 255, 255))
        pygame.gfxdraw.aacircle(self.screen, int(center.x), int(center.y), 8, (255, 255, 255))

    def _render_ui(self):
        # Level
        level_text = self.font_main.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=self.SCREEN_WIDTH / 2, top=10)
        self.screen.blit(score_text, score_rect)

        # Timer
        time_left = max(0, self.VICTORY_TIME_SECONDS - self.timer)
        timer_text = self.font_timer.render(f"{time_left:.2f}", True, self.COLOR_TIMER)
        timer_rect = timer_text.get_rect(right=self.SCREEN_WIDTH - 10, top=10)
        self.screen.blit(timer_text, timer_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play ---
    env = GameEnv()
    obs, info = env.reset()
    
    # Override Pygame display for manual playing
    # This check is needed because os.environ is set for headless mode
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    
    pygame.display.init()
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fractal Survival")

    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        movement_action = 0  # No-op
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement_action = 2
        
        if keys[pygame.K_SPACE]:
            space_action = 1
            
        if keys[pygame.K_r]: # Reset
            obs, info = env.reset()
            total_reward = 0

        action = [movement_action, space_action, 0] # Shift is unused
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        truncated = trunc
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(env.screen, frame)
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
    env.close()