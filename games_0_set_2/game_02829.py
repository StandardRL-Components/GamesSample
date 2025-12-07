import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to accelerate. Pick up passengers (red) and drop them off at their destination (yellow)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade taxi game. Race against the clock to earn $1000 by picking up and delivering fares. Drive fast, but don't crash!"
    )

    # Frames auto-advance at 30fps.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 120
        self.WIN_SCORE = 1000

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_ROAD = (40, 40, 45)
        self.COLOR_BUILDING = (50, 60, 70)
        self.COLOR_BUILDING_OUTLINE = (80, 90, 100)
        self.COLOR_TAXI = (80, 220, 100)
        self.COLOR_TAXI_OUTLINE = (40, 110, 50)
        self.COLOR_FARE = (255, 50, 50)
        self.COLOR_DESTINATION = (255, 255, 0)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_UI_BG = (10, 10, 10, 180)

        # Physics and game parameters
        self.ACCELERATION = 0.4
        self.MAX_SPEED = 6.0
        self.DRAG = 0.95
        self.CRASH_BOUNCE = -0.75
        self.PICKUP_RADIUS = 25
        self.PARTICLE_LIFETIME = 20

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.buildings = []
        self.road_areas = []
        self.has_fare = None
        self.fare_pos = None
        self.destination_building_idx = None
        self.score = None
        self.time_remaining = None
        self.game_over = None
        self.steps = None
        self.particles = []
        self.last_dist_to_target = 0

        # This will be called once in reset() to create a consistent map
        self.map_generated = False

        # self.reset() is called by the environment wrapper, no need to call it here.
        # self.validate_implementation() # This can be called by a test script, not in __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)

        if not self.map_generated:
            self._generate_map()
            self.map_generated = True

        self.has_fare = False
        self.destination_building_idx = None
        self._spawn_fare()

        self.score = 0
        self.time_remaining = self.GAME_DURATION_SECONDS * self.FPS
        self.game_over = False
        self.steps = 0
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        # Calculate reward based on distance change to target
        old_dist_to_target = self._get_dist_to_target()
        
        self._update_player_state(movement)
        self._update_game_state()

        new_dist_to_target = self._get_dist_to_target()
        distance_delta = old_dist_to_target - new_dist_to_target
        
        # Continuous reward for moving towards the target
        reward = distance_delta * 0.02

        # Event-based rewards
        reward += self._handle_collisions_and_events()

        # Update timer and steps
        self.time_remaining -= 1
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0:
            reward -= 100
            terminated = True
            self.game_over = True

        truncated = False # No truncation condition in this game
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player_state(self, movement):
        # Update velocity based on action
        accel = np.array([0.0, 0.0])
        if movement == 1: accel[1] = -self.ACCELERATION # UP
        elif movement == 2: accel[1] = self.ACCELERATION # DOWN
        elif movement == 3: accel[0] = -self.ACCELERATION # LEFT
        elif movement == 4: accel[0] = self.ACCELERATION # RIGHT
        
        self.player_vel += accel
        self.player_vel *= self.DRAG

        # Clamp speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.MAX_SPEED:
            self.player_vel = (self.player_vel / speed) * self.MAX_SPEED

        # Update position
        self.player_pos += self.player_vel

    def _handle_collisions_and_events(self):
        event_reward = 0

        # World wrapping
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

        # Building collisions
        player_rect = pygame.Rect(self.player_pos[0] - 8, self.player_pos[1] - 8, 16, 16)
        for building in self.buildings:
            if player_rect.colliderect(building):
                # Simple push-back and bounce
                overlap_x = (player_rect.width / 2 + building.width / 2) - abs(player_rect.centerx - building.centerx)
                overlap_y = (player_rect.height / 2 + building.height / 2) - abs(player_rect.centery - building.centery)

                if overlap_x < overlap_y:
                    if player_rect.centerx < building.centerx:
                        self.player_pos[0] -= overlap_x
                    else:
                        self.player_pos[0] += overlap_x
                    self.player_vel[0] *= self.CRASH_BOUNCE
                else:
                    if player_rect.centery < building.centery:
                        self.player_pos[1] -= overlap_y
                    else:
                        self.player_pos[1] += overlap_y
                    self.player_vel[1] *= self.CRASH_BOUNCE
                
                event_reward -= 1.0 # Crash penalty
                # Sound effect: Crash!
                break

        # Fare pickup
        if not self.has_fare and self.fare_pos is not None:
            dist_to_fare = np.linalg.norm(self.player_pos - self.fare_pos)
            if dist_to_fare < self.PICKUP_RADIUS:
                self.has_fare = True
                self.fare_pos = None
                self._spawn_destination()
                fare_value = 50 + self.np_random.integers(0, 51)
                self.score += fare_value
                event_reward += 10
                # Sound effect: Cha-ching!

        # Fare drop-off
        if self.has_fare and self.destination_building_idx is not None:
            dest_building = self.buildings[self.destination_building_idx]
            if dest_building.collidepoint(self.player_pos):
                self.has_fare = False
                self.destination_building_idx = None
                self._spawn_fare()
                fare_value = 100 + self.np_random.integers(0, 101)
                self.score += fare_value
                event_reward += 20
                # Sound effect: Success!
        
        return event_reward

    def _update_game_state(self):
        # Add particles for tire tracks
        if np.linalg.norm(self.player_vel) > 1.5:
            self.particles.append([self.player_pos.copy(), self.PARTICLE_LIFETIME])
        
        # Update and remove old particles
        self.particles = [p for p in self.particles if p[1] > 0]
        for p in self.particles:
            p[1] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_ROAD)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for pos, lifetime in self.particles:
            alpha = int(255 * (lifetime / self.PARTICLE_LIFETIME))
            color = (max(0, self.COLOR_ROAD[0] + alpha // 4), max(0, self.COLOR_ROAD[1] + alpha // 4), max(0, self.COLOR_ROAD[2] + alpha // 4))
            pygame.draw.circle(self.screen, color, pos.astype(int), 2)
            
        # Render buildings
        for building in self.buildings:
            pygame.draw.rect(self.screen, self.COLOR_BUILDING, building)
            pygame.draw.rect(self.screen, self.COLOR_BUILDING_OUTLINE, building, 1)

        # Render destination
        if self.has_fare and self.destination_building_idx is not None:
            dest_building = self.buildings[self.destination_building_idx]
            pulse = abs(math.sin(self.steps * 0.1))
            size_increase = int(10 * pulse)
            glow_rect = dest_building.inflate(size_increase, size_increase)
            
            # Use gfxdraw for anti-aliased filled rectangle
            color = (*self.COLOR_DESTINATION, int(80 + 60 * pulse))
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
            
            pygame.draw.rect(self.screen, self.COLOR_DESTINATION, dest_building, 3, border_radius=3)

        # Render fare
        if not self.has_fare and self.fare_pos is not None:
            pulse = abs(math.sin(self.steps * 0.2))
            radius = int(self.PICKUP_RADIUS * 0.5 + pulse * 5)
            alpha = int(150 + pulse * 105)
            pygame.gfxdraw.filled_circle(self.screen, int(self.fare_pos[0]), int(self.fare_pos[1]), radius, (*self.COLOR_FARE, alpha))
            pygame.gfxdraw.aacircle(self.screen, int(self.fare_pos[0]), int(self.fare_pos[1]), radius, self.COLOR_FARE)

        # Render player taxi
        player_size = 16
        player_rect = pygame.Rect(0, 0, player_size, player_size)
        player_rect.center = self.player_pos
        pygame.draw.rect(self.screen, self.COLOR_TAXI_OUTLINE, player_rect.inflate(4, 4), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_TAXI, player_rect, border_radius=3)

    def _render_ui(self):
        # UI Background
        ui_surf = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        # Score
        score_text = self.font_small.render(f"SCORE: ${self.score} / ${self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_str = f"TIME: {max(0, int(self.time_remaining / self.FPS))}"
        time_text = self.font_small.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Objective Arrow
        target_pos = self._get_target_pos()
        if target_pos is not None:
            angle_rad = math.atan2(target_pos[1] - self.player_pos[1], target_pos[0] - self.player_pos[0])
            arrow_dist = 30
            arrow_end_x = self.player_pos[0] + arrow_dist * math.cos(angle_rad)
            arrow_end_y = self.player_pos[1] + arrow_dist * math.sin(angle_rad)
            
            arrow_color = self.COLOR_DESTINATION if self.has_fare else self.COLOR_FARE
            p1 = (arrow_end_x, arrow_end_y)
            p2 = (arrow_end_x - 8 * math.cos(angle_rad + 0.5), arrow_end_y - 8 * math.sin(angle_rad + 0.5))
            p3 = (arrow_end_x - 8 * math.cos(angle_rad - 0.5), arrow_end_y - 8 * math.sin(angle_rad - 0.5))
            pygame.draw.polygon(self.screen, arrow_color, [p1, p2, p3])

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME'S UP!"
            color = self.COLOR_TAXI if self.score >= self.WIN_SCORE else self.COLOR_FARE
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
        }

    def _generate_map(self):
        self.buildings.clear()
        self.road_areas.clear()
        
        # Use a fixed seed for consistent map generation
        map_rng = random.Random(42)
        
        grid_w, grid_h = 10, 6
        cell_w, cell_h = self.WIDTH / grid_w, self.HEIGHT / grid_h
        
        for r in range(grid_h):
            for c in range(grid_w):
                # Chance to place a building
                if map_rng.random() < 0.4:
                    b_w = map_rng.randint(int(cell_w * 0.6), int(cell_w * 0.9))
                    b_h = map_rng.randint(int(cell_h * 0.6), int(cell_h * 0.9))
                    b_x = c * cell_w + (cell_w - b_w) / 2
                    b_y = r * cell_h + (cell_h - b_h) / 2
                    self.buildings.append(pygame.Rect(b_x, b_y, b_w, b_h))
                else:
                    self.road_areas.append(pygame.Rect(c*cell_w, r*cell_h, cell_w, cell_h))

    def _get_valid_spawn_point(self):
        while True:
            # FIX: self.np_random.choice converts the list of pygame.Rect objects
            # into a 2D numpy array. A row from this array (a 1D numpy array)
            # is returned, which does not have .left/.right attributes.
            # Instead, we randomly select an index and then get the Rect object.
            idx = self.np_random.integers(len(self.road_areas))
            area = self.road_areas[idx]
            
            x = self.np_random.uniform(area.left, area.right)
            y = self.np_random.uniform(area.top, area.bottom)
            # Ensure it's not too close to the edge of the area
            if area.width > 20 and area.height > 20:
                if area.left + 10 < x < area.right - 10 and area.top + 10 < y < area.bottom - 10:
                    return np.array([x, y])

    def _spawn_fare(self):
        self.fare_pos = self._get_valid_spawn_point()

    def _spawn_destination(self):
        self.destination_building_idx = self.np_random.integers(len(self.buildings))

    def _get_target_pos(self):
        if self.has_fare and self.destination_building_idx is not None:
            return np.array(self.buildings[self.destination_building_idx].center)
        elif not self.has_fare and self.fare_pos is not None:
            return self.fare_pos
        return None

    def _get_dist_to_target(self):
        target_pos = self._get_target_pos()
        if target_pos is not None:
            return np.linalg.norm(self.player_pos - target_pos)
        return 0

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup Pygame window for human play
    # Unset the dummy video driver to allow a window to be created
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crazy Taxi Gym")
    clock = pygame.time.Clock()

    total_reward = 0
    
    # Game loop
    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
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

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Rendering ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate Control ---
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()