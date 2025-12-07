import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:08:18.082607
# Source Brief: brief_00284.md
# Brief Index: 284
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Pilot a drone through a dangerous asteroid field to collect valuable minerals. "
        "Use the minerals to upgrade your drone's thrusters and survive as long as possible."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your drone. "
        "Press space to purchase thruster upgrades when you have enough minerals."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # Colors
    COLOR_BG = (10, 10, 26)
    COLOR_DRONE = (0, 255, 128)
    COLOR_DRONE_GLOW = (0, 255, 128, 64)
    COLOR_TRAIL = (0, 255, 128)
    COLOR_ASTEROID = (128, 128, 148)
    COLOR_MINERAL = (255, 255, 0)
    COLOR_MINERAL_GLOW = (255, 255, 0, 80)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_UI_VALUE = (255, 255, 255)
    COLOR_UPGRADE_READY = (0, 180, 255)

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_upgrade = pygame.font.Font(None, 32)
        
        # --- Internal State ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0.0

        self.drone_pos = None
        self.drone_vel = None
        self.drone_trail = None
        
        self.minerals_collected = 0
        self.upgrade_level = 0
        self.upgrade_cost = 0

        self.asteroids = []
        self.minerals = []
        self.stars = []
        
        self.space_pressed_last_frame = False
        self.last_density_increase_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Drone
        self.drone_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.drone_vel = pygame.math.Vector2(0, 0)
        self.drone_trail = []
        
        # Economy & Progression
        self.minerals_collected = 0
        self.upgrade_level = 0
        self.upgrade_cost = 20
        
        # Other state
        self.space_pressed_last_frame = False
        self.last_density_increase_step = 0

        # --- Procedural Generation ---
        player_start_avoid_radius = 150
        self._spawn_stars(150)
        self.asteroids = self._spawn_asteroids(10, self.drone_pos, player_start_avoid_radius)
        self.minerals = self._spawn_minerals(8, self.drone_pos, player_start_avoid_radius)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = 0.0
        
        if self.game_over:
            # If the game is already over, do nothing but return terminal state
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input & Update State ---
        self._handle_input(action)
        self._update_game_state()
        self._handle_collisions()
        
        # --- Calculate Reward & Termination ---
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        truncated = False # This game ends on a terminal condition, not a step limit per se
        if terminated and not self.game_over: # Survived the full time
            reward += 50.0
            self.score += 50.0
        elif self.game_over: # Collided with asteroid
            reward = -100.0
            self.score += -100.0 # Overwrite score with final penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Movement ---
        thrust = 0.25 * (1 + self.upgrade_level * 0.3)
        if movement == 1: self.drone_vel.y -= thrust # Up
        if movement == 2: self.drone_vel.y += thrust # Down
        if movement == 3: self.drone_vel.x -= thrust # Left
        if movement == 4: self.drone_vel.x += thrust # Right

        # --- Upgrade Purchase ---
        if space_held and not self.space_pressed_last_frame:
            if self.minerals_collected >= self.upgrade_cost:
                # SFX: upgrade_purchased.wav
                self.minerals_collected -= self.upgrade_cost
                self.upgrade_level += 1
                self.upgrade_cost = int(20 + self.upgrade_level * 15)
                self.reward_this_step += 5.0
        self.space_pressed_last_frame = space_held

    def _update_game_state(self):
        self.steps += 1
        
        # --- Drone Physics ---
        drag = 0.98
        self.drone_vel *= drag
        if self.drone_vel.length() > 8: # Speed cap
            self.drone_vel.scale_to_length(8)
        self.drone_pos += self.drone_vel
        
        # World wrap
        self.drone_pos.x %= self.SCREEN_WIDTH
        self.drone_pos.y %= self.SCREEN_HEIGHT
        
        # Update Trail
        self.drone_trail.append(self.drone_pos.copy())
        if len(self.drone_trail) > 15:
            self.drone_trail.pop(0)
            
        # --- Update Entities ---
        for asteroid in self.asteroids:
            asteroid['angle'] += asteroid['rot_speed']
        
        # --- Difficulty Scaling ---
        # Increase asteroid density every 10 seconds (300 steps)
        if self.steps > 0 and self.steps % 300 == 0 and self.steps != self.last_density_increase_step:
            self.last_density_increase_step = self.steps
            new_asteroids = self._spawn_asteroids(2, self.drone_pos, 100)
            self.asteroids.extend(new_asteroids)

    def _handle_collisions(self):
        # Drone vs Minerals
        for mineral in self.minerals[:]:
            dist = self.drone_pos.distance_to(mineral['pos'])
            if dist < 10 + mineral['size']: # Drone radius ~10
                # SFX: mineral_collect.wav
                self.minerals.remove(mineral)
                self.minerals_collected += 5
                self.reward_this_step += 1.0
                # Spawn a new mineral somewhere else
                self.minerals.extend(self._spawn_minerals(1))

        # Drone vs Asteroids
        for asteroid in self.asteroids:
            dist = self.drone_pos.distance_to(asteroid['pos'])
            if dist < 10 + asteroid['size'] * 0.8: # Use 80% of size for forgiving hitbox
                # SFX: explosion.wav
                self.game_over = True
                break

    def _calculate_reward(self):
        # Survival reward is given for every step
        return 0.1 + self.reward_this_step
    
    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

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
            "minerals": self.minerals_collected,
            "upgrade_level": self.upgrade_level,
        }

    # --- Rendering Methods ---

    def _render_game(self):
        self._render_background()
        self._render_minerals()
        self._render_asteroids()
        self._render_drone()

    def _render_background(self):
        for star in self.stars:
            # Parallax effect
            star_pos = (
                (star['pos'][0] - self.drone_pos.x * star['depth']) % self.SCREEN_WIDTH,
                (star['pos'][1] - self.drone_pos.y * star['depth']) % self.SCREEN_HEIGHT
            )
            pygame.draw.circle(self.screen, star['color'], star_pos, star['size'])
    
    def _render_minerals(self):
        for mineral in self.minerals:
            self._draw_wrapped(self._draw_pulsating_mineral, mineral)

    def _draw_pulsating_mineral(self, pos, mineral):
        # Pulsating effect for glow
        glow_alpha = 60 + math.sin(self.steps * 0.1 + mineral['phase']) * 40
        glow_size = mineral['size'] + 2 + math.sin(self.steps * 0.1 + mineral['phase']) * 2
        
        # Use gfxdraw for anti-aliasing
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(glow_size), (*self.COLOR_MINERAL, int(glow_alpha)))
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(glow_size), (*self.COLOR_MINERAL, int(glow_alpha)))
        
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), mineral['size'], self.COLOR_MINERAL)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), mineral['size'], self.COLOR_MINERAL)


    def _render_asteroids(self):
        for asteroid in self.asteroids:
            self._draw_wrapped(self._draw_rotating_asteroid, asteroid)

    def _draw_rotating_asteroid(self, pos, asteroid):
        points = []
        for i in range(asteroid['num_vertices']):
            angle = 2 * math.pi * i / asteroid['num_vertices'] + math.radians(asteroid['angle'])
            x = pos.x + math.cos(angle) * asteroid['size'] * asteroid['shape'][i]
            y = pos.y + math.sin(angle) * asteroid['size'] * asteroid['shape'][i]
            points.append((int(x), int(y)))
        
        if len(points) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_drone(self):
        # Trail
        if len(self.drone_trail) > 2:
            for i in range(len(self.drone_trail) - 1):
                alpha = int(255 * (i / len(self.drone_trail)))
                color = (*self.COLOR_TRAIL, alpha)
                start_pos = self.drone_trail[i]
                end_pos = self.drone_trail[i+1]
                pygame.draw.line(self.screen, color, start_pos, end_pos, max(1, int(i/5)))

        # Drone Body
        drone_points = [
            pygame.math.Vector2(0, -10),
            pygame.math.Vector2(-6, 8),
            pygame.math.Vector2(6, 8)
        ]
        
        # Glow effect
        glow_size = 18 + math.sin(self.steps * 0.2) * 3
        pygame.gfxdraw.filled_circle(self.screen, int(self.drone_pos.x), int(self.drone_pos.y), int(glow_size), self.COLOR_DRONE_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(self.drone_pos.x), int(self.drone_pos.y), int(glow_size), self.COLOR_DRONE_GLOW)
        
        # Rotate points based on velocity direction for a nice touch
        angle = self.drone_vel.angle_to(pygame.math.Vector2(0, -1))
        rotated_points = [p.rotate(-angle) + self.drone_pos for p in drone_points]

        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_DRONE)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_DRONE)

    def _render_ui(self):
        # --- Time Remaining ---
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # --- Mineral Count ---
        mineral_text = self.font_ui.render("MINERALS:", True, self.COLOR_UI_TEXT)
        mineral_value = self.font_ui.render(f"{self.minerals_collected}", True, self.COLOR_MINERAL)
        self.screen.blit(mineral_text, (10, 10))
        self.screen.blit(mineral_value, (mineral_text.get_width() + 15, 10))
        
        # --- Upgrade Level & Cost ---
        color = self.COLOR_UPGRADE_READY if self.minerals_collected >= self.upgrade_cost else self.COLOR_UI_VALUE
        upgrade_text = self.font_upgrade.render(f"THRUST LVL: {self.upgrade_level + 1}", True, self.COLOR_UI_VALUE)
        cost_text = self.font_upgrade.render(f"UPGRADE COST: {self.upgrade_cost}", True, color)
        
        upgrade_pos_x = (self.SCREEN_WIDTH - upgrade_text.get_width()) / 2
        cost_pos_x = (self.SCREEN_WIDTH - cost_text.get_width()) / 2
        self.screen.blit(upgrade_text, (upgrade_pos_x, self.SCREEN_HEIGHT - 60))
        self.screen.blit(cost_text, (cost_pos_x, self.SCREEN_HEIGHT - 35))

    def _draw_wrapped(self, draw_func, entity):
        """Draws an entity, handling screen wrapping by drawing it multiple times if near an edge."""
        pos = entity['pos']
        size = entity.get('size', 20) # Use a default size if not present
        
        # Normal draw
        draw_func(pos, entity)
        
        # Wrapped draw
        draw_x, draw_y = False, False
        wrapped_pos = pos.copy()

        if pos.x < size: # Near left edge
            wrapped_pos.x = pos.x + self.SCREEN_WIDTH
            draw_x = True
        elif pos.x > self.SCREEN_WIDTH - size: # Near right edge
            wrapped_pos.x = pos.x - self.SCREEN_WIDTH
            draw_x = True

        if pos.y < size: # Near top edge
            wrapped_pos.y = pos.y + self.SCREEN_HEIGHT
            draw_y = True
        elif pos.y > self.SCREEN_HEIGHT - size: # Near bottom edge
            wrapped_pos.y = pos.y - self.SCREEN_HEIGHT
            draw_y = True
            
        if draw_x:
            draw_func(pygame.math.Vector2(wrapped_pos.x, pos.y), entity)
        if draw_y:
            draw_func(pygame.math.Vector2(pos.x, wrapped_pos.y), entity)
        if draw_x and draw_y:
            draw_func(wrapped_pos, entity)

    # --- Spawning Methods ---

    def _spawn_stars(self, count):
        self.stars = []
        for _ in range(count):
            depth = random.uniform(0.1, 0.5)
            self.stars.append({
                'pos': (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                'size': random.uniform(0.5, 1.5),
                'color': (random.randint(100, 200), random.randint(100, 200), random.randint(200, 255)),
                'depth': depth
            })

    def _spawn_asteroids(self, count, avoid_pos=None, avoid_radius=0):
        asteroids = []
        for _ in range(count):
            while True:
                pos = pygame.math.Vector2(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT))
                if avoid_pos is None or pos.distance_to(avoid_pos) > avoid_radius:
                    break
            
            num_vertices = random.randint(5, 9)
            asteroids.append({
                'pos': pos,
                'size': random.randint(20, 45),
                'rot_speed': random.uniform(-1.0, 1.0),
                'angle': random.uniform(0, 360),
                'num_vertices': num_vertices,
                'shape': [random.uniform(0.7, 1.0) for _ in range(num_vertices)]
            })
        return asteroids

    def _spawn_minerals(self, count, avoid_pos=None, avoid_radius=0):
        minerals = []
        for _ in range(count):
            while True:
                pos = pygame.math.Vector2(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT))
                if avoid_pos is None or pos.distance_to(avoid_pos) > avoid_radius:
                    break
            minerals.append({
                'pos': pos,
                'size': random.randint(5, 8),
                'phase': random.uniform(0, 2 * math.pi)
            })
        return minerals

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    # The main loop expects a display, so we need to create one.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc. depending on your OS
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Drone Miner")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    
    done = False
    total_score = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move")
    print("Space: Purchase Upgrade")
    print("R: Reset")
    print("Q: Quit")
    
    while not done:
        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_score = 0
                    print("--- Game Reset ---")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for R or Q
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        waiting = False
                        done = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_score = 0
                        print("--- Game Reset ---")
                        waiting = False

        clock.tick(GameEnv.FPS)
        
    env.close()