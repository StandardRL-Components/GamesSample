import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a unicycle/motorbike.
    The goal is to collect 100 coins without falling over or running out of coins.
    Falling deducts a large number of coins.
    The player can transform between a nimble unicycle and a fast but unstable motorbike.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up/accelerate, 2=down/brake, 3=left/lean left, 4=right/lean right)
    - actions[1]: Space button (0=released, 1=held) for transformation
    - actions[2]: Shift button (0=released, 1=held) - unused

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.1 per coin collected
    - +0.01 per step while moving fast and balanced
    - -0.1 for leaning too far
    - +1 for transforming
    - -1 for falling
    - +100 for winning (100 coins)
    - -50 for losing (score < 0)
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = (
        "Ride a transforming unicycle-motorbike, maintain your balance, and collect 100 coins to win."
    )
    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to lean. Press space to transform between unicycle and motorbike."
    )
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 18, 48)
    COLOR_GROUND = (69, 40, 60)
    COLOR_STARS = (200, 200, 220)
    COLOR_COIN = (255, 223, 0)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_SPEED_BAR_BG = (50, 50, 80)
    COLOR_SPEED_BAR_FG = (100, 120, 255)
    
    # Player Colors
    COLOR_UNI_BODY = (220, 50, 50)
    COLOR_UNI_WHEEL = (50, 50, 50)
    COLOR_BIKE_BODY = (50, 150, 220)
    COLOR_BIKE_WHEEL = (60, 60, 60)

    # Game Parameters
    MAX_STEPS = 2000
    WIN_SCORE = 100
    GROUND_Y = 350
    
    # Physics Parameters
    GRAVITY = 0.3
    FALL_ANGLE = math.pi / 2  # 90 degrees
    WARN_ANGLE = math.pi / 4.5 # 40 degrees
    LEAN_TORQUE = 0.015
    ANGULAR_DAMPING = 0.95
    
    # Unicycle Physics
    UNI_ACCELERATION = 0.15
    UNI_COM_HEIGHT = 60 # Center of Mass height
    UNI_FRICTION = 0.99
    
    # Motorbike Physics
    BIKE_ACCELERATION = 0.3
    BIKE_COM_HEIGHT = 75 # Higher CoM = less stable
    BIKE_FRICTION = 0.995

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.stars = [] # Will be initialized in reset
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.player_world_x = 0.0
        self.player_vx = 0.0
        self.player_angle = 0.0
        self.player_v_angle = 0.0
        self.is_motorbike = False
        self.is_fallen = False
        self.fall_timer = 0
        self.space_was_held = False
        self.camera_x = 0.0
        self.coins = deque()
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 25 # Start with some coins
        
        # Player state
        self.player_world_x = 0.0
        self.player_vx = 0.0
        self.player_angle = 0.0
        self.player_v_angle = 0.0
        self.is_motorbike = False
        self.is_fallen = False
        self.fall_timer = 0
        
        # Action state
        self.space_was_held = False
        
        # World state
        self.camera_x = 0.0
        self.coins = deque()
        self.particles = []
        self._spawn_initial_coins()
        
        if not self.stars:
            self.stars = [(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.GROUND_Y - 50)) for _ in range(100)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = 0

        self.steps += 1
        
        # Handle falling state
        if self.is_fallen:
            self.fall_timer -= 1
            if self.fall_timer <= 0:
                self._recover_from_fall()
        else:
            # --- Handle Actions ---
            reward += self._handle_transformation(space_held)
            self._apply_player_controls(movement)

        # --- Update Physics ---
        self._update_physics()

        # --- Update World and Check Interactions ---
        self._manage_coins()
        reward += self._check_coin_collection()
        
        # --- Manage Particles ---
        self._update_particles()
        
        # --- Calculate Step Rewards ---
        if not self.is_fallen:
            if abs(self.player_vx) > 5 and abs(self.player_angle) < self.WARN_ANGLE / 2:
                reward += 0.01 # Reward for stable high-speed movement
            if abs(self.player_angle) > self.WARN_ANGLE:
                reward -= 0.1 # Penalty for being unstable
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.score < 0:
                reward += -50 # Lose penalty

        self.space_was_held = bool(space_held)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_transformation(self, space_held):
        if space_held and not self.space_was_held:
            self.is_motorbike = not self.is_motorbike
            # Sound: "transform.wav"
            # Add particles for transformation effect
            player_screen_pos = (self.SCREEN_WIDTH // 2, self.GROUND_Y - 30)
            color = self.COLOR_BIKE_BODY if self.is_motorbike else self.COLOR_UNI_BODY
            for _ in range(50):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                p_vx = math.cos(angle) * speed
                p_vy = math.sin(angle) * speed
                self.particles.append([list(player_screen_pos), [p_vx, p_vy], self.np_random.integers(20, 41), color])
            return 1.0 # Reward for transforming
        return 0.0

    def _apply_player_controls(self, movement):
        accel = self.BIKE_ACCELERATION if self.is_motorbike else self.UNI_ACCELERATION
        
        # Lean
        if movement == 3: # Left
            self.player_v_angle -= self.LEAN_TORQUE
        elif movement == 4: # Right
            self.player_v_angle += self.LEAN_TORQUE
            
        # Accelerate/Brake
        if movement == 1: # Up
            self.player_vx += accel
        elif movement == 2: # Down
            self.player_vx -= accel

    def _update_physics(self):
        # --- Angular Physics (Balance) ---
        if not self.is_fallen:
            com_height = self.BIKE_COM_HEIGHT if self.is_motorbike else self.UNI_COM_HEIGHT
            # Torque from gravity trying to pull the player down
            gravity_torque = math.sin(self.player_angle) * self.GRAVITY * (com_height / 50.0)
            self.player_v_angle += gravity_torque
        
        # Apply angular velocity and damping
        self.player_v_angle *= self.ANGULAR_DAMPING
        self.player_angle += self.player_v_angle / (60 / 30) # Adjust for FPS

        # --- Horizontal Physics (Movement) ---
        friction = self.BIKE_FRICTION if self.is_motorbike else self.UNI_FRICTION
        self.player_vx *= friction
        self.player_world_x += self.player_vx
        self.camera_x = self.player_world_x
        
        # --- Check for Fall ---
        if not self.is_fallen and abs(self.player_angle) > self.FALL_ANGLE:
            self._fall()
            
        # --- Speed particles ---
        if abs(self.player_vx) > 1:
            # Sound: "whoosh.wav" (at low volume)
            for _ in range(int(abs(self.player_vx) / 4)):
                y = self.np_random.integers(self.GROUND_Y - 100, self.GROUND_Y)
                self.particles.append([[self.SCREEN_WIDTH, y], [-15 - self.player_vx, 0], 20, (100, 100, 120)])

    def _fall(self):
        # Sound: "crash.wav"
        self.is_fallen = True
        self.fall_timer = 60 # Cooldown period of 2 seconds (at 30fps)
        self.score -= 20
        self.player_vx = 0
        
        # Add fall sparks
        player_screen_pos = (self.SCREEN_WIDTH // 2, self.GROUND_Y - 30)
        for _ in range(30):
            angle = self.np_random.uniform(-math.pi, 0)
            speed = self.np_random.uniform(1, 7)
            p_vx = math.cos(angle) * speed
            p_vy = math.sin(angle) * speed
            # FIX: Lifetime was up to 50, causing alpha > 255. Capped at 40.
            self.particles.append([list(player_screen_pos), [p_vx, p_vy], self.np_random.integers(30, 41), (255, 255, 200)])
        return -1.0 # Reward for falling

    def _recover_from_fall(self):
        self.is_fallen = False
        self.player_angle = 0
        self.player_v_angle = 0

    def _check_coin_collection(self):
        collected_reward = 0
        player_hitbox_x = self.player_world_x
        for coin in list(self.coins):
            coin_x, _ = coin
            if abs(coin_x - player_hitbox_x) < 20:
                # Sound: "coin.wav"
                self.coins.remove(coin)
                self.score += 1
                collected_reward += 0.1
                # Add coin collection particles
                coin_screen_x = int(coin_x - self.camera_x + self.SCREEN_WIDTH / 2)
                for _ in range(15):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    p_vx = math.cos(angle) * speed
                    p_vy = math.sin(angle) * speed
                    self.particles.append([[coin_screen_x, self.GROUND_Y - 20], [p_vx, p_vy], self.np_random.integers(15, 31), self.COLOR_COIN])
        return collected_reward

    def _spawn_initial_coins(self):
        for i in range(20):
            self.coins.append((300 + i * 150 + self.np_random.uniform(-30, 30), self.GROUND_Y - 20))

    def _manage_coins(self):
        # Remove coins far behind the player
        while self.coins and self.coins[0][0] < self.camera_x - self.SCREEN_WIDTH:
            self.coins.popleft()
        # Add new coins far ahead of the player
        while len(self.coins) < 30:
            last_coin_x = self.coins[-1][0] if self.coins else self.player_world_x
            new_coin_x = last_coin_x + 150 + self.np_random.uniform(-30, 30)
            self.coins.append((new_coin_x, self.GROUND_Y - 20))

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        if self.score < 0:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "is_motorbike": self.is_motorbike,
            "player_vx": self.player_vx,
            "player_angle": self.player_angle,
        }
        
    def _render_game(self):
        # --- Background ---
        for x, y in self.stars:
            screen_x = (x - self.camera_x * 0.1) % self.SCREEN_WIDTH
            pygame.draw.circle(self.screen, self.COLOR_STARS, (int(screen_x), int(y)), 1)
        
        # --- Ground ---
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
        
        # --- Coins ---
        for coin_x, coin_y in self.coins:
            screen_x = int(coin_x - self.camera_x + self.SCREEN_WIDTH / 2)
            if 0 < screen_x < self.SCREEN_WIDTH:
                # Animate coin bobbing
                bob = math.sin(self.steps * 0.1 + coin_x * 0.1) * 3
                pygame.gfxdraw.filled_circle(self.screen, screen_x, int(coin_y + bob), 10, self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, screen_x, int(coin_y + bob), 10, self.COLOR_COIN)

        # --- Particles ---
        for pos, _, lifetime, color in self.particles:
            # The alpha calculation assumes a max lifetime of 40.
            alpha = int(255 * (lifetime / 40))
            alpha = max(0, min(255, alpha)) # Clamp to be safe
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(s, color + (alpha,), (2, 2), 2)
            self.screen.blit(s, (int(pos[0]), int(pos[1])))
        
        # --- Player ---
        self._render_player()

    def _render_player(self):
        player_screen_x = self.SCREEN_WIDTH // 2
        player_screen_y = self.GROUND_Y
        
        angle = -self.player_angle if not self.is_fallen else -self.FALL_ANGLE * np.sign(self.player_angle)

        if self.is_motorbike:
            self._render_vehicle(player_screen_x, player_screen_y, angle, self.COLOR_BIKE_BODY, self.COLOR_BIKE_WHEEL, is_bike=True)
        else:
            self._render_vehicle(player_screen_x, player_screen_y, angle, self.COLOR_UNI_BODY, self.COLOR_UNI_WHEEL, is_bike=False)
            
    def _render_vehicle(self, cx, cy, angle, body_color, wheel_color, is_bike):
        # Glow effect
        for i in range(15, 0, -2):
            glow_alpha = 40 - i * 2
            pygame.gfxdraw.filled_circle(self.screen, cx, cy - 30, 40 + i, body_color + (glow_alpha,))

        # Helper to rotate points
        def rotate(p, angle_rad):
            x, y = p
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            return x * cos_a - y * sin_a, x * sin_a + y * cos_a

        if is_bike:
            wheel_dist = 35
            # Define points relative to center (0,0)
            points = {
                'w1': (-wheel_dist, 0), 'w2': (wheel_dist, 0),
                'seat': (0, -40), 'handle': (25, -50),
                'engine': (0, -15), 'frame1': (-wheel_dist, -20),
                'frame2': (wheel_dist, -20)
            }
            
            # Rotate all points
            rotated_points = {k: rotate(v, angle) for k, v in points.items()}
            
            # Draw wheels
            w1_pos = (int(cx + rotated_points['w1'][0]), int(cy - 25 + rotated_points['w1'][1]))
            w2_pos = (int(cx + rotated_points['w2'][0]), int(cy - 25 + rotated_points['w2'][1]))
            pygame.gfxdraw.filled_circle(self.screen, w1_pos[0], w1_pos[1], 20, wheel_color)
            pygame.gfxdraw.filled_circle(self.screen, w2_pos[0], w2_pos[1], 20, wheel_color)
            
            # Draw frame
            seat_pos = (int(cx + rotated_points['seat'][0]), int(cy - 25 + rotated_points['seat'][1]))
            handle_pos = (int(cx + rotated_points['handle'][0]), int(cy - 25 + rotated_points['handle'][1]))
            engine_pos = (int(cx + rotated_points['engine'][0]), int(cy - 25 + rotated_points['engine'][1]))
            f1_pos = (int(cx + rotated_points['frame1'][0]), int(cy - 25 + rotated_points['frame1'][1]))
            f2_pos = (int(cx + rotated_points['frame2'][0]), int(cy - 25 + rotated_points['frame2'][1]))

            # FIX: Use pygame.draw.line to specify width, not aaline.
            pygame.draw.line(self.screen, body_color, w1_pos, seat_pos, 3)
            pygame.draw.line(self.screen, body_color, w2_pos, seat_pos, 3)
            pygame.draw.line(self.screen, body_color, seat_pos, handle_pos, 3)
            pygame.draw.line(self.screen, body_color, engine_pos, f1_pos, 3)
            pygame.draw.line(self.screen, body_color, engine_pos, f2_pos, 3)
            pygame.gfxdraw.filled_circle(self.screen, engine_pos[0], engine_pos[1], 10, body_color)
        else: # Unicycle
            # Define points relative to wheel center
            points = {'seat': (0, -60), 'pedal_axle': (0, 0)}
            rotated_points = {k: rotate(v, angle) for k, v in points.items()}
            
            # Draw wheel
            wheel_pos = (cx, cy - 20)
            pygame.gfxdraw.filled_circle(self.screen, wheel_pos[0], wheel_pos[1], 20, wheel_color)
            pygame.gfxdraw.aacircle(self.screen, wheel_pos[0], wheel_pos[1], 20, (80,80,80))
            
            # Draw frame
            seat_pos = (int(wheel_pos[0] + rotated_points['seat'][0]), int(wheel_pos[1] + rotated_points['seat'][1]))
            axle_pos = (int(wheel_pos[0] + rotated_points['pedal_axle'][0]), int(wheel_pos[1] + rotated_points['pedal_axle'][1]))
            pygame.draw.line(self.screen, body_color, axle_pos, seat_pos, 5)
            pygame.draw.line(self.screen, body_color, (seat_pos[0] - 20, seat_pos[1]), (seat_pos[0] + 20, seat_pos[1]), 5)

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_ui.render(f"COINS: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # --- Speed Bar ---
        max_speed = 25
        speed_fraction = min(1, abs(self.player_vx) / max_speed)
        bar_width = 150
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_SPEED_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_SPEED_BAR_FG, (bar_x, bar_y, bar_width * speed_fraction, bar_height))
        
        # --- Win/Loss Message ---
        if self.score >= self.WIN_SCORE:
            msg_text = self.font_msg.render("YOU WIN!", True, self.COLOR_COIN)
            text_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_text, text_rect)
        elif self.score < 0:
            msg_text = self.font_msg.render("GAME OVER", True, self.COLOR_UNI_BODY)
            text_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Control Mapping ---
    # ARROW KEYS for movement/lean
    # SPACE to transform
    
    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    # Main game loop
    running = True
    
    # Create a window for human rendering
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Game Test")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0
            
        # Transform
        if keys[pygame.K_SPACE]:
            action[1] = 1
        else:
            action[1] = 0
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Use a separate Pygame window for human rendering
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(render_surface, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()