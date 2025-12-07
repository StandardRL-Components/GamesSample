import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:06:24.697393
# Source Brief: brief_02583.md
# Brief Index: 2583
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the player guides a group of accelerating bouncing balls 
    to hit all targets before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a group of bouncing balls to hit all targets before time runs out. "
        "Apply directional force to influence their chaotic movement."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to apply a directional force to all balls."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 60

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_BALLS = [
        (255, 87, 34), (3, 169, 244), (255, 235, 59), (76, 175, 80),
        (233, 30, 99), (0, 188, 212), (255, 193, 7), (139, 195, 74),
        (156, 39, 176), (63, 81, 181)
    ]
    COLOR_TARGET = (255, 255, 255)
    COLOR_OBSTACLE = (50, 50, 70)
    COLOR_UI_TEXT = (220, 220, 220)

    # Game Parameters
    NUM_BALLS = 10
    BALL_RADIUS = 4
    BALL_INITIAL_SPEED = 2.0
    BALL_ACCEL_ON_BOUNCE = 1.05
    PLAYER_VELOCITY_CHANGE = 0.4

    NUM_TARGETS = 20
    TARGET_RADIUS = 6

    # Rewards
    REWARD_HIT_TARGET = 10.0
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0
    REWARD_CLOSER_TO_TARGET = 0.1
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False

        self.balls = []
        self.targets = []
        self.obstacles = []
        self.particles = []

        self.last_avg_dist_to_target = float('inf')

        # --- Initial Reset ---
        # The first reset is done here to initialize state for validation if needed
        # self.reset() # Avoid calling reset in init
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION_SECONDS * self.FPS

        # --- Clear Dynamic Elements ---
        self.balls.clear()
        self.targets.clear()
        self.obstacles.clear()
        self.particles.clear()

        # --- Create Obstacles ---
        self._create_obstacles()

        # --- Create Targets ---
        self._create_targets()

        # --- Create Balls ---
        self._create_balls()

        self.last_avg_dist_to_target = self._calculate_avg_dist_to_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement, _, _ = action # space_held and shift_held are unused
        
        # --- Update Game Clock ---
        self.steps += 1
        self.time_remaining -= 1
        
        # --- Initialize Step Reward ---
        reward = 0

        # --- Apply Player Action ---
        self._apply_player_action(movement)

        # --- Update Game Logic ---
        self._update_balls()
        reward += self._check_target_collisions()
        self._update_particles()
        
        # --- Continuous Reward ---
        current_avg_dist = self._calculate_avg_dist_to_target()
        if current_avg_dist < self.last_avg_dist_to_target:
            reward += self.REWARD_CLOSER_TO_TARGET
        self.last_avg_dist_to_target = current_avg_dist

        # --- Check Termination Conditions ---
        terminated = False
        if not self.targets: # Win condition
            self.game_over = True
            terminated = True
            reward += self.REWARD_WIN
        elif self.time_remaining <= 0: # Loss condition
            self.game_over = True
            terminated = True
            reward += self.REWARD_LOSS

        # Per Gymnasium API, truncated is for time limits, not game-over states
        truncated = self.time_remaining <= 0 and not self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Helper Methods for State Initialization ---

    def _create_obstacles(self):
        self.obstacles.append(pygame.Rect(100, 150, 150, 20))
        self.obstacles.append(pygame.Rect(390, 150, 150, 20))
        self.obstacles.append(pygame.Rect(270, 50, 20, 100))
        self.obstacles.append(pygame.Rect(270, 250, 20, 100))

    def _create_targets(self):
        while len(self.targets) < self.NUM_TARGETS:
            pos = pygame.Vector2(
                self.np_random.uniform(self.TARGET_RADIUS, self.SCREEN_WIDTH - self.TARGET_RADIUS),
                self.np_random.uniform(self.TARGET_RADIUS, self.SCREEN_HEIGHT - self.TARGET_RADIUS)
            )
            
            # Check for overlap with other targets and obstacles
            valid_pos = True
            for obs in self.obstacles:
                if obs.collidepoint(pos):
                    valid_pos = False
                    break
            if not valid_pos: continue

            for target in self.targets:
                if pos.distance_to(target['pos']) < self.TARGET_RADIUS * 2:
                    valid_pos = False
                    break
            if not valid_pos: continue
            
            self.targets.append({'pos': pos})

    def _create_balls(self):
        for i in range(self.NUM_BALLS):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS),
                    self.np_random.uniform(self.BALL_RADIUS, self.SCREEN_HEIGHT - self.BALL_RADIUS)
                )
                
                # Ensure balls don't spawn inside obstacles
                valid_pos = True
                for obs in self.obstacles:
                    if obs.collidepoint(pos):
                        valid_pos = False
                        break
                if valid_pos:
                    break
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_INITIAL_SPEED
            
            self.balls.append({
                'pos': pos,
                'vel': vel,
                'prev_pos': pos.copy(),
                'color': self.COLOR_BALLS[i % len(self.COLOR_BALLS)]
            })

    # --- Helper Methods for Step Logic ---

    def _apply_player_action(self, movement):
        vel_change = pygame.Vector2(0, 0)
        if movement == 1: vel_change.y = -self.PLAYER_VELOCITY_CHANGE  # Up
        elif movement == 2: vel_change.y = self.PLAYER_VELOCITY_CHANGE   # Down
        elif movement == 3: vel_change.x = -self.PLAYER_VELOCITY_CHANGE  # Left
        elif movement == 4: vel_change.x = self.PLAYER_VELOCITY_CHANGE   # Right

        for ball in self.balls:
            ball['vel'] += vel_change

    def _update_balls(self):
        for ball in self.balls:
            ball['prev_pos'] = ball['pos'].copy()
            ball['pos'] += ball['vel']
            self._handle_ball_collisions(ball)

    def _handle_ball_collisions(self, ball):
        bounced = False
        # Wall collisions
        if ball['pos'].x < self.BALL_RADIUS or ball['pos'].x > self.SCREEN_WIDTH - self.BALL_RADIUS:
            ball['vel'].x *= -1
            ball['pos'].x = np.clip(ball['pos'].x, self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            bounced = True
        if ball['pos'].y < self.BALL_RADIUS or ball['pos'].y > self.SCREEN_HEIGHT - self.BALL_RADIUS:
            ball['vel'].y *= -1
            ball['pos'].y = np.clip(ball['pos'].y, self.BALL_RADIUS, self.SCREEN_HEIGHT - self.BALL_RADIUS)
            bounced = True

        # Obstacle collisions
        for obs in self.obstacles:
            if obs.collidepoint(ball['pos']):
                # Simple but effective bounce logic based on previous position
                prev_rect = pygame.Rect(ball['prev_pos'].x - self.BALL_RADIUS, ball['prev_pos'].y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
                
                if prev_rect.right <= obs.left or prev_rect.left >= obs.right:
                    ball['vel'].x *= -1
                if prev_rect.bottom <= obs.top or prev_rect.top >= obs.bottom:
                    ball['vel'].y *= -1

                # Move ball out of obstacle to prevent sticking
                ball['pos'] = ball['prev_pos'].copy()
                bounced = True
                break # Assume one obstacle collision per frame is enough
        
        if bounced:
            ball['vel'] *= self.BALL_ACCEL_ON_BOUNCE
            # Sound effect placeholder
            # pygame.mixer.Sound("bounce.wav").play()
    
    def _check_target_collisions(self):
        hit_reward = 0
        targets_to_remove = []
        for i, target in enumerate(self.targets):
            for ball in self.balls:
                if ball['pos'].distance_to(target['pos']) < self.BALL_RADIUS + self.TARGET_RADIUS:
                    if target not in targets_to_remove:
                        targets_to_remove.append(target)
                        hit_reward += self.REWARD_HIT_TARGET
                        self.score += 1
                        # Sound effect placeholder
                        # pygame.mixer.Sound("hit.wav").play()
                        self._create_particle_effect(target['pos'], self.COLOR_TARGET)
                        break
        
        if targets_to_remove:
            self.targets = [t for t in self.targets if t not in targets_to_remove]
        
        return hit_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['radius'] += p['growth_rate']
            p['alpha'] = max(0, int(255 * (p['life'] / p['max_life'])))

    def _create_particle_effect(self, pos, color):
        self.particles.append({
            'pos': pos,
            'radius': self.TARGET_RADIUS,
            'max_radius': self.TARGET_RADIUS * 3,
            'growth_rate': 0.5,
            'color': color,
            'life': 30, # 0.5 seconds at 60fps
            'max_life': 30,
            'alpha': 255
        })

    def _calculate_avg_dist_to_target(self):
        if not self.targets or not self.balls:
            return 0
        
        total_min_dist = 0
        for ball in self.balls:
            min_dist = min(ball['pos'].distance_to(t['pos']) for t in self.targets)
            total_min_dist += min_dist
        
        return total_min_dist / len(self.balls)

    # --- Rendering and Info Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)

        # Render targets
        for target in self.targets:
            pos = (int(target['pos'].x), int(target['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.TARGET_RADIUS, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.TARGET_RADIUS, self.COLOR_TARGET)
        
        # Render particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            color = (*p['color'], p['alpha'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Render balls
        for ball in self.balls:
            pos = (int(ball['pos'].x), int(ball['pos'].y))
            # Glow effect
            glow_color = (*ball['color'], 50)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 3, glow_color)
            # Main ball
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, ball['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, ball['color'])

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left_sec = max(0, self.time_remaining // self.FPS)
        timer_text = self.font.render(f"TIME: {time_left_sec}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "targets_left": len(self.targets)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Pygame window for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bouncing Balls Environment")
    
    total_reward = 0
    
    while not terminated and not truncated:
        # --- Human Input ---
        movement_action = 0 # 0=none
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4

        # The action space is MultiDiscrete, but we only use the first part for manual control
        action = [movement_action, 0, 0]

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render for Display ---
        # The observation is (H, W, C), but pygame needs a surface.
        # We can just re-render to the display screen.
        render_screen.blit(env.screen, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()