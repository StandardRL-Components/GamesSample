import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:48:42.517359
# Source Brief: brief_00038.md
# Brief Index: 38
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a bicycle.
    The goal is to collect coins to reach a target score while maintaining balance
    and avoiding obstacles, all within a time limit.

    **Visuals:**
    - A side-scrolling view with a parallax background.
    - The player's bicycle leans and moves based on physics.
    - Bright, high-contrast elements for gameplay clarity.
    - A dedicated UI shows score, time, and a balance meter.

    **Gameplay:**
    - The agent must balance the bike by leaning left and right.
    - Pedaling increases speed, which makes balancing easier but also increases risk.
    - Coins grant points and a positive reward.
    - Obstacles are deadly and must be avoided.
    - The episode ends on victory (score goal), failure (fall, crash, time out).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Balance a bicycle, pedal to gain speed, and collect coins while avoiding obstacles. "
        "Reach the target score before time runs out or you fall over!"
    )
    user_guide = "Controls: ↑ to pedal, ↓ to brake. Use ← and → to lean and maintain your balance."
    auto_advance = True

    # --- Constants ---
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 350
    WORLD_BOUNDS_PADDING = 200

    # Colors
    COLOR_SKY = (135, 206, 235)
    COLOR_GROUND = (107, 142, 35)
    COLOR_HILL_1 = (127, 162, 55) # Darker, further back
    COLOR_HILL_2 = (147, 182, 75) # Lighter, closer
    COLOR_BICYCLE = (20, 20, 20)
    COLOR_RIDER = (217, 87, 99)
    COLOR_COIN = (255, 215, 0)
    COLOR_OBSTACLE = (139, 0, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_BALANCE_SAFE = (0, 200, 0)
    COLOR_BALANCE_WARN = (255, 165, 0)
    COLOR_BALANCE_DANGER = (255, 0, 0)

    # Game Rules
    WIN_SCORE = 75
    MAX_STEPS = 4500  # 45 seconds at ~100 FPS internal clock
    FALL_ANGLE_DEGREES = 50

    # Physics
    GRAVITY_PULL = 0.0025
    LEAN_INPUT_FORCE = 0.003
    LEAN_DAMPING = 0.98
    STEERING_CORRECTION_FACTOR = 0.00015
    PEDAL_FORCE = 0.04
    BRAKE_FORCE = 0.06
    FRICTION = 0.995
    MAX_SPEED = 5.0
    
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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.render_mode = render_mode

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_x = 0
        self.bike_pos = pygame.Vector2(0, 0)
        self.bike_velocity = 0.0
        self.bike_lean_angle = 0.0
        self.bike_lean_velocity = 0.0
        self.coins = []
        self.obstacles = []
        self.particles = []
        self.hills = []
        self.next_obstacle_x = 0
        self.next_coin_x = 0
        self.last_reward = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_reward = 0

        # Bicycle state
        self.camera_x = 0.0
        self.bike_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.GROUND_Y)
        self.bike_velocity = 1.0 # Start with some speed
        self.bike_lean_angle = 0.0
        self.bike_lean_velocity = 0.0

        # World elements
        self.particles = []
        self._generate_initial_world()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- 1. Update Game Logic based on Action ---
        self._update_player(action)
        self._update_world()

        # --- 2. Calculate Reward ---
        reward = self._calculate_reward()
        self.last_reward = reward

        # --- 3. Check for Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 15 # Fall/loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, action):
        movement = action[0]

        # --- Handle player input ---
        player_lean_input = 0
        if movement == 3:  # Lean Left
            player_lean_input = -self.LEAN_INPUT_FORCE
        elif movement == 4:  # Lean Right
            player_lean_input = self.LEAN_INPUT_FORCE

        # --- Update lean physics ---
        # Gravity's effect on lean
        gravity_torque = -math.sin(math.radians(self.bike_lean_angle)) * self.GRAVITY_PULL
        
        # Steering correction: faster speed makes it easier to balance
        steering_correction_torque = math.sin(math.radians(self.bike_lean_angle)) * self.bike_velocity * self.STEERING_CORRECTION_FACTOR
        
        self.bike_lean_velocity += player_lean_input + gravity_torque - steering_correction_torque
        self.bike_lean_velocity *= self.LEAN_DAMPING
        self.bike_lean_angle += self.bike_lean_velocity

        # --- Update forward motion ---
        if movement == 1:  # Pedal Forward
            self.bike_velocity += self.PEDAL_FORCE
        elif movement == 2:  # Pedal Backward / Brake
            self.bike_velocity -= self.BRAKE_FORCE
        
        self.bike_velocity *= self.FRICTION
        self.bike_velocity = max(0, min(self.bike_velocity, self.MAX_SPEED))
        
        self.camera_x += self.bike_velocity

    def _update_world(self):
        # --- Collision Detection: Coins ---
        for coin in self.coins[:]:
            if self._get_bike_hitbox().colliderect(coin):
                self.coins.remove(coin)
                self.score += 5
                # Sfx: coin_pickup.wav
                for _ in range(15):
                    self.particles.append(Particle(coin.center, self.COLOR_COIN, self.np_random))

        # --- Collision Detection: Obstacles ---
        for obstacle in self.obstacles:
            if self._get_bike_hitbox().colliderect(obstacle):
                self.game_over = True
                # Sfx: crash.wav
                for _ in range(30):
                    self.particles.append(Particle(self.bike_pos, self.COLOR_OBSTACLE, self.np_random))
                break

        # --- Update Particles ---
        for p in self.particles[:]:
            p.update()
            if p.is_dead():
                self.particles.remove(p)

        # --- Procedural Generation ---
        self._cleanup_entities()
        self._spawn_entities()

    def _calculate_reward(self):
        # Continuous rewards to encourage good behavior
        reward = 0
        # Reward for staying upright
        reward += 0.1 * (1 - abs(self.bike_lean_angle) / self.FALL_ANGLE_DEGREES)
        # Reward for moving forward
        reward += 0.2 * (self.bike_velocity / self.MAX_SPEED)
        return reward

    def _check_termination(self):
        # Win condition
        if self.score >= self.WIN_SCORE:
            return True
        # Loss conditions
        if abs(self.bike_lean_angle) > self.FALL_ANGLE_DEGREES:
            return True
        # The game_over flag from obstacle collision is checked here
        if self.game_over:
            return True
        return False

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / 100,
            "velocity": self.bike_velocity,
            "lean_angle": self.bike_lean_angle,
        }
        
    def _get_bike_hitbox(self):
        # A simple rectangle for collision
        return pygame.Rect(self.bike_pos.x - 15, self.bike_pos.y - 40, 30, 40)

    # --- Rendering Methods ---

    def _render_all(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()

    def _render_background(self):
        self.screen.fill(self.COLOR_SKY)
        
        # Parallax hills
        for i, (x_offset, height, width, color, speed) in enumerate(self.hills):
            hill_x = (x_offset - self.camera_x * speed) % (self.SCREEN_WIDTH + width) - width
            pygame.draw.ellipse(self.screen, color, [hill_x, self.GROUND_Y - height, width, height])

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, [0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y])

    def _render_game_elements(self):
        # Coins
        for coin in self.coins:
            screen_x = coin.x - self.camera_x
            if 0 < screen_x < self.SCREEN_WIDTH:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_x + coin.width/2), int(coin.y + coin.height/2), int(coin.width/2), self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, int(screen_x + coin.width/2), int(coin.y + coin.height/2), int(coin.width/2), (255,255,255))


        # Obstacles
        for obstacle in self.obstacles:
            screen_x = obstacle.x - self.camera_x
            if 0 < screen_x < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, [screen_x, obstacle.y, obstacle.width, obstacle.height])

        # Bicycle
        self._draw_bicycle()

        # Particles
        for p in self.particles:
            p.draw(self.screen, self.camera_x)

    def _draw_bicycle(self):
        # Create a temporary surface to draw the bike on, for easy rotation
        bike_surf = pygame.Surface((100, 100), pygame.SRCALPHA)
        pivot = (50, 80) # Pivot point for rotation (bottom bracket)

        # Rider
        rider_rect = pygame.Rect(0, 0, 12, 25)
        rider_rect.center = (pivot[0], pivot[1] - 25)
        pygame.draw.ellipse(bike_surf, self.COLOR_RIDER, rider_rect)
        
        # Frame
        handlebar = (pivot[0] + 20, pivot[1] - 25)
        seat = (pivot[0] - 15, pivot[1] - 20)
        rear_axle = (pivot[0] - 25, pivot[1])
        front_axle = (pivot[0] + 25, pivot[1])
        
        pygame.draw.line(bike_surf, self.COLOR_BICYCLE, seat, handlebar, 3)
        pygame.draw.line(bike_surf, self.COLOR_BICYCLE, pivot, seat, 3)
        pygame.draw.line(bike_surf, self.COLOR_BICYCLE, pivot, rear_axle, 3)
        pygame.draw.line(bike_surf, self.COLOR_BICYCLE, pivot, front_axle, 3)
        pygame.draw.line(bike_surf, self.COLOR_BICYCLE, seat, rear_axle, 3)
        pygame.draw.line(bike_surf, self.COLOR_BICYCLE, handlebar, front_axle, 3)

        # Wheels
        pygame.gfxdraw.filled_circle(bike_surf, int(front_axle[0]), int(front_axle[1]), 12, self.COLOR_BICYCLE)
        pygame.gfxdraw.filled_circle(bike_surf, int(rear_axle[0]), int(rear_axle[1]), 12, self.COLOR_BICYCLE)
        pygame.gfxdraw.filled_circle(bike_surf, int(front_axle[0]), int(front_axle[1]), 8, self.COLOR_SKY)
        pygame.gfxdraw.filled_circle(bike_surf, int(rear_axle[0]), int(rear_axle[1]), 8, self.COLOR_SKY)

        # Rotate and blit
        rotated_bike = pygame.transform.rotate(bike_surf, self.bike_lean_angle)
        new_rect = rotated_bike.get_rect(center=(self.bike_pos.x, self.bike_pos.y - 40))
        self.screen.blit(rotated_bike, new_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 100)
        timer_text = self.font_large.render(f"Time: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Balance Meter
        self._draw_balance_meter()
        
        # Game Over Text
        if self.game_over:
            outcome_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (0, 255, 0) if self.score >= self.WIN_SCORE else (255, 0, 0)
            go_text = self.font_large.render(outcome_text, True, color)
            go_rect = go_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            pygame.draw.rect(self.screen, self.COLOR_UI_BG, go_rect.inflate(20, 20))
            self.screen.blit(go_text, go_rect)

    def _draw_balance_meter(self):
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 25
        radius = 50
        
        # Draw the background arcs
        for angle in range(-90, 91, 5):
            rad = math.radians(angle - 90)
            start_point = (center_x + radius * math.cos(rad), center_y + radius * math.sin(rad))
            end_point = (center_x + (radius+10) * math.cos(rad), center_y + (radius+10) * math.sin(rad))
            
            percent_angle = abs(angle) / self.FALL_ANGLE_DEGREES
            color = self.COLOR_BALANCE_SAFE
            if percent_angle > 0.6: color = self.COLOR_BALANCE_WARN
            if percent_angle > 0.9: color = self.COLOR_BALANCE_DANGER
            pygame.draw.line(self.screen, color, start_point, end_point, 3)

        # Draw the needle
        needle_angle_rad = math.radians(self.bike_lean_angle - 90)
        needle_end = (center_x + (radius+5) * math.cos(needle_angle_rad), center_y + (radius+5) * math.sin(needle_angle_rad))
        pygame.draw.line(self.screen, (255, 255, 255), (center_x, center_y), needle_end, 3)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 5, (255,255,255))


    # --- World Generation ---

    def _generate_initial_world(self):
        self.coins = []
        self.obstacles = []
        self.hills = []
        
        # Generate background hills
        for _ in range(10): # Further, slower hills
            self.hills.append((self.np_random.integers(0, self.SCREEN_WIDTH * 2), self.np_random.integers(50, 150), self.np_random.integers(300, 600), self.COLOR_HILL_1, 0.2))
        for _ in range(5): # Closer, faster hills
            self.hills.append((self.np_random.integers(0, self.SCREEN_WIDTH * 2), self.np_random.integers(20, 80), self.np_random.integers(200, 400), self.COLOR_HILL_2, 0.5))

        self.next_obstacle_x = self.camera_x + self.SCREEN_WIDTH
        self.next_coin_x = self.camera_x + self.SCREEN_WIDTH / 2
        self._spawn_entities()

    def _spawn_entities(self):
        # Spawn obstacles
        obstacle_freq = 500 - (self.steps / self.MAX_STEPS) * 400 # Freq increases from 500 to 100
        if self.next_obstacle_x < self.camera_x + self.SCREEN_WIDTH + self.WORLD_BOUNDS_PADDING:
            spawn_x = self.next_obstacle_x
            self.obstacles.append(pygame.Rect(spawn_x, self.GROUND_Y - 20, 30, 20))
            self.next_obstacle_x += self.np_random.integers(obstacle_freq, obstacle_freq + 200)

        # Spawn coins
        coin_freq = 100
        if self.next_coin_x < self.camera_x + self.SCREEN_WIDTH + self.WORLD_BOUNDS_PADDING:
            spawn_x = self.next_coin_x
            # Spawn a cluster of coins
            for i in range(self.np_random.integers(2, 5)):
                y_offset = self.np_random.choice([0, -40, -80])
                self.coins.append(pygame.Rect(spawn_x + i * 30, self.GROUND_Y - 40 + y_offset, 20, 20))
            self.next_coin_x += self.np_random.integers(coin_freq, coin_freq + 150)

    def _cleanup_entities(self):
        # Remove entities that are far behind the camera
        self.coins = [c for c in self.coins if c.x > self.camera_x - self.WORLD_BOUNDS_PADDING]
        self.obstacles = [o for o in self.obstacles if o.x > self.camera_x - self.WORLD_BOUNDS_PADDING]

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()

class Particle:
    def __init__(self, pos, color, rng):
        self.pos = pygame.Vector2(pos)
        self.color = color
        self.rng = rng
        self.vel = pygame.Vector2(rng.uniform(-2, 2), rng.uniform(-4, 0))
        self.lifespan = rng.integers(20, 40)
        self.size = rng.integers(3, 7)

    def update(self):
        self.pos += self.vel
        self.vel.y += 0.1  # Gravity
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

    def is_dead(self):
        return self.lifespan <= 0

    def draw(self, surface, camera_x):
        screen_x = self.pos.x - camera_x
        pygame.draw.circle(surface, self.color, (int(screen_x), int(self.pos.y)), int(self.size))


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Balance Bike")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    # Game loop for manual play
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT or keys[pygame.K_q]:
                done = True

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # obs, info = env.reset() # Uncomment to auto-reset
            # total_reward = 0
            done = True # End after one episode for this script

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Limit FPS for smooth manual play

    pygame.quit()