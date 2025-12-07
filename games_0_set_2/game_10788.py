import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:56:51.355773
# Source Brief: brief_00788.md
# Brief Index: 788
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent pilots a drone to deliver packages in a procedurally generated city.
    The agent must navigate traffic, which increases with each successful delivery. The goal is to deliver
    15 packages within the time limit.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    # --- FIX: Added required class attributes ---
    game_description = (
        "Pilot a drone to deliver packages in a procedurally generated city, avoiding buildings and increasing traffic."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to fly the drone. Press 'space' to pick up a package or make a delivery."
    )
    auto_advance = True
    # --- END FIX ---

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # --- Visuals & Colors ---
        self.COLOR_BG = (26, 28, 44)          # Dark blue-grey
        self.COLOR_BUILDING = (40, 44, 52)    # Lighter grey
        self.COLOR_PLAYER = (0, 255, 255)     # Cyan
        self.COLOR_PLAYER_GLOW = (0, 128, 128)
        self.COLOR_PACKAGE = (255, 255, 0)    # Yellow
        self.COLOR_PACKAGE_GLOW = (128, 128, 0)
        self.COLOR_DELIVERY = (0, 255, 0)     # Green
        self.COLOR_DELIVERY_GLOW = (0, 128, 0)
        self.COLOR_TRAFFIC = (255, 65, 54)    # Red
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_COLLISION = (255, 0, 0, 100) # Semi-transparent red

        self.font_main = pygame.font.Font(None, 32)
        self.font_indicator = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Game Constants ---
        self.TARGET_DELIVERIES = 15
        self.TIME_LIMIT_SECONDS = 300
        self.PLAYER_SPEED = 5
        self.PLAYER_SIZE = 12
        self.PACKAGE_SIZE = 10
        self.DELIVERY_SIZE = 20
        self.PICKUP_RADIUS = 30
        self.COLLISION_PENALTY_SECONDS = 5
        self.INITIAL_TRAFFIC_COUNT = 5

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.deliveries_made = 0
        self.player_pos = None
        self.player_trail = None
        self.has_package = False
        self.package_pos = None
        self.delivery_pos = None
        self.buildings = []
        self.traffic = []
        self.last_space_held = False
        self.collision_flash_timer = 0
        self.last_dist_to_target = float('inf')


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.metadata["render_fps"]
        self.deliveries_made = 0
        
        self.player_pos = pygame.Vector2(self.screen_width / 2, self.screen_height / 2)
        self.player_trail = deque(maxlen=15)
        self.has_package = False
        
        self.buildings = self._generate_buildings()
        self.package_pos = self._spawn_location(self.PACKAGE_SIZE)
        self.delivery_pos = self._spawn_location(self.DELIVERY_SIZE)
        
        self.traffic = []
        for _ in range(self.INITIAL_TRAFFIC_COUNT):
            self._add_traffic_unit()

        self.last_space_held = False
        self.collision_flash_timer = 0
        self.last_dist_to_target = self.player_pos.distance_to(self.package_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Update Game Logic ---
        self._update_player(movement)
        self._update_traffic()
        
        # --- Handle Interactions ---
        interaction_reward = self._handle_interactions(space_held)
        reward += interaction_reward

        # --- Time and Collision ---
        self.time_remaining -= 1
        if self.collision_flash_timer > 0:
            self.collision_flash_timer -= 1
            
        collision_reward = self._check_collisions()
        reward += collision_reward

        # --- Reward Shaping ---
        current_target = self.delivery_pos if self.has_package else self.package_pos
        dist_to_target = self.player_pos.distance_to(current_target)
        if dist_to_target < self.last_dist_to_target:
            reward += 0.01 # Small reward for getting closer
        else:
            reward -= 0.01 # Small penalty for moving away
        self.last_dist_to_target = dist_to_target

        # --- Termination Check ---
        terminated = False
        win = self.deliveries_made >= self.TARGET_DELIVERIES
        loss = self.time_remaining <= 0

        if win:
            reward += 100.0
            terminated = True
            # Sound: game_win.wav
        elif loss:
            reward -= 100.0
            terminated = True
            # Sound: game_over.wav
        
        if terminated:
            self.game_over = True

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos.x += self.PLAYER_SPEED
        
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.screen_width)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.screen_height)
        
        self.player_trail.append(self.player_pos.copy())

    def _update_traffic(self):
        for car in self.traffic:
            car['rect'].move_ip(car['vel'])
            # Wrap around screen
            if car['rect'].right < 0: car['rect'].left = self.screen_width
            if car['rect'].left > self.screen_width: car['rect'].right = 0
            if car['rect'].bottom < 0: car['rect'].top = self.screen_height
            if car['rect'].top > self.screen_height: car['rect'].bottom = 0

    def _handle_interactions(self, space_held):
        reward = 0
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        if space_pressed:
            if not self.has_package:
                if self.player_pos.distance_to(self.package_pos) < self.PICKUP_RADIUS:
                    self.has_package = True
                    # Sound: package_pickup.wav
                    self.delivery_pos = self._spawn_location(self.DELIVERY_SIZE)
                    self.last_dist_to_target = self.player_pos.distance_to(self.delivery_pos)
            else:
                if self.player_pos.distance_to(self.delivery_pos) < self.PICKUP_RADIUS:
                    self.has_package = False
                    self.score += 10
                    self.deliveries_made += 1
                    reward += 10.0
                    # Sound: package_delivery.wav
                    self.package_pos = self._spawn_location(self.PACKAGE_SIZE)
                    self.last_dist_to_target = self.player_pos.distance_to(self.package_pos)
                    # Increase difficulty
                    self._add_traffic_unit()
        return reward

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Check traffic collision
        for car in self.traffic:
            if player_rect.colliderect(car['rect']):
                if self.collision_flash_timer <= 0: # Prevent repeated penalties
                    self.time_remaining -= self.COLLISION_PENALTY_SECONDS * self.metadata["render_fps"]
                    self.collision_flash_timer = 15 # frames
                    # Sound: collision.wav
                    return -0.5
        return 0.0

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
            "deliveries_made": self.deliveries_made,
            "time_remaining_sec": self.time_remaining / self.metadata["render_fps"]
        }

    def _render_game(self):
        # Buildings
        for building in self.buildings:
            pygame.draw.rect(self.screen, self.COLOR_BUILDING, building)

        # Delivery point
        if self.has_package:
            self._draw_glowing_circle(self.delivery_pos, self.DELIVERY_SIZE, self.COLOR_DELIVERY, self.COLOR_DELIVERY_GLOW)

        # Package
        if not self.has_package:
            self._draw_glowing_circle(self.package_pos, self.PACKAGE_SIZE, self.COLOR_PACKAGE, self.COLOR_PACKAGE_GLOW)

        # Traffic
        for car in self.traffic:
            pygame.draw.rect(self.screen, self.COLOR_TRAFFIC, car['rect'])

        # Player Trail
        if len(self.player_trail) > 1:
            for i, pos in enumerate(self.player_trail):
                alpha = int(255 * (i / len(self.player_trail)))
                trail_color = (*self.COLOR_PLAYER, alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(self.PLAYER_SIZE/2 * (i/len(self.player_trail))), trail_color)

        # Player
        self._draw_glowing_circle(self.player_pos, self.PLAYER_SIZE, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        
        # Collision flash
        if self.collision_flash_timer > 0:
            flash_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_COLLISION)
            self.screen.blit(flash_surface, (0,0))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"DELIVERIES: {self.deliveries_made}/{self.TARGET_DELIVERIES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_sec = max(0, self.time_remaining // self.metadata["render_fps"])
        mins, secs = divmod(time_sec, 60)
        timer_text = self.font_main.render(f"TIME: {int(mins):02d}:{int(secs):02d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.screen_width - timer_text.get_width() - 10, 10))

        # Destination Indicator
        if not self.game_over:
            target_pos, target_color, text = (0,0), (0,0,0), ""
            if self.has_package:
                target_pos, target_color, text = self.delivery_pos, self.COLOR_DELIVERY, "DELIVER TO GREEN ZONE"
            else:
                target_pos, target_color, text = self.package_pos, self.COLOR_PACKAGE, "COLLECT YELLOW PACKAGE"
            
            indicator_text = self.font_indicator.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(indicator_text, (self.screen_width/2 - indicator_text.get_width()/2, self.screen_height - 30))

            # Arrow
            angle = math.atan2(target_pos.y - self.player_pos.y, target_pos.x - self.player_pos.x)
            arrow_points = []
            for i in range(3):
                a = angle + (i - 1) * math.pi / 2.5
                p = self.player_pos + pygame.Vector2(math.cos(a), math.sin(a)) * (self.PLAYER_SIZE + 10)
                arrow_points.append((int(p.x), int(p.y)))
            pygame.draw.polygon(self.screen, target_color, arrow_points)

        # Game Over Message
        if self.game_over:
            msg = "MISSION COMPLETE" if self.deliveries_made >= self.TARGET_DELIVERIES else "MISSION FAILED"
            color = self.COLOR_DELIVERY if self.deliveries_made >= self.TARGET_DELIVERIES else self.COLOR_TRAFFIC
            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(game_over_text, text_rect)

    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        # Draw a series of concentric, transparent circles for the glow effect
        for i in range(radius, 0, -2):
            alpha = 100 * (1 - i / radius)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius + i, (*glow_color, alpha))
        # Draw the main circle
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, color)

    def _generate_buildings(self, count=10, min_size=(40, 40), max_size=(100, 100)):
        buildings = []
        for _ in range(count):
            for _ in range(100): # Max 100 attempts to place a building
                w = self.np_random.integers(min_size[0], max_size[0])
                h = self.np_random.integers(min_size[1], max_size[1])
                x = self.np_random.integers(0, self.screen_width - w)
                y = self.np_random.integers(0, self.screen_height - h)
                new_building = pygame.Rect(x, y, w, h)
                if not any(new_building.colliderect(b) for b in buildings):
                    buildings.append(new_building)
                    break
        return buildings
    
    def _is_valid_location(self, rect):
        if rect.left < 0 or rect.right > self.screen_width or rect.top < 0 or rect.bottom > self.screen_height:
            return False
        for building in self.buildings:
            if rect.colliderect(building):
                return False
        return True

    def _spawn_location(self, size):
        for _ in range(100): # Max 100 attempts
            x = self.np_random.integers(size, self.screen_width - size)
            y = self.np_random.integers(size, self.screen_height - size)
            rect = pygame.Rect(x-size/2, y-size/2, size, size)
            if self._is_valid_location(rect):
                return pygame.Vector2(x, y)
        return pygame.Vector2(self.screen_width / 2, self.screen_height / 2) # Fallback

    def _add_traffic_unit(self):
        for _ in range(100): # Max 100 attempts
            w, h = (self.np_random.integers(20, 40), self.np_random.integers(10, 15))
            vel = pygame.Vector2(0, 0)
            
            if self.np_random.choice([True, False]): # Horizontal or Vertical
                # Horizontal
                speed = self.np_random.uniform(1.5, 3.0) * self.np_random.choice([-1, 1])
                vel.x = speed
                x = -w if speed > 0 else self.screen_width
                y = self.np_random.integers(0, self.screen_height - h)
            else:
                # Vertical
                w, h = h, w # Swap dimensions for vertical traffic
                speed = self.np_random.uniform(1.5, 3.0) * self.np_random.choice([-1, 1])
                vel.y = speed
                y = -h if speed > 0 else self.screen_height
                x = self.np_random.integers(0, self.screen_width - w)
            
            rect = pygame.Rect(x, y, w, h)
            if self._is_valid_location(rect):
                self.traffic.append({'rect': rect, 'vel': vel})
                break

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display window
    pygame.display.set_caption("Drone Delivery")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move")
    print("Space:  Pickup/Deliver")
    print("Q:      Quit")
    
    while not done:
        # --- Action Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
        
        env.clock.tick(env.metadata["render_fps"])

    print(f"\nGame Over!")
    print(f"Final Info: {info}")
    print(f"Total Reward: {total_reward:.2f}")
    
    env.close()