import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:40:15.366046
# Source Brief: brief_00601.md
# Brief Index: 601
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Cyberpunk Runner Gymnasium Environment
    
    The agent controls a character escaping corporate drones in a neon cityscape.
    The goal is to survive as long as possible by collecting energy and avoiding drones.
    A combo system rewards chaining actions like boosting and collecting items quickly.
    
    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Magnetic Boost (0=released, 1=pressed)
    - actions[2]: Unused (0=released, 1=pressed)

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - an RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Escape corporate drones in a neon cityscape. Survive as long as possible by collecting energy, "
        "avoiding detection, and using your boost."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move. Press space to activate a speed boost at the cost of energy."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (16, 5, 32) # #100520
    COLOR_PLAYER = (0, 255, 255) # #00FFFF
    COLOR_DRONE = (255, 0, 51) # #FF0033
    COLOR_CAMERA = (157, 0, 255) # #9D00FF
    COLOR_ENERGY = (0, 255, 102) # #00FF66
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR = (0, 150, 255)
    COLOR_UI_BAR_BG = (50, 50, 80)

    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 5.0
    PLAYER_FRICTION = 0.9
    PLAYER_MAX_ENERGY = 100
    PLAYER_ENERGY_DECAY = 0.05 # Per step
    BOOST_COST = 15
    BOOST_SPEED = 15.0
    BOOST_DURATION = 5 # steps

    # Drones
    DRONE_SIZE = 10
    DRONE_INITIAL_COUNT = 3
    DRONE_MAX_COUNT = 8
    DRONE_SPAWN_INTERVAL = 150 # steps
    DRONE_BASE_SPEED = 1.0
    DRONE_SPEED_INCREASE_RATE = 0.05
    DRONE_SPEED_INCREASE_INTERVAL = 100
    
    # Cameras
    CAMERA_COUNT = 2
    CAMERA_RADIUS = 70
    CAMERA_ALARM_DURATION = 90 # steps
    CAMERA_ALARM_SPEED_MULTIPLIER = 2.0

    # Resources
    ENERGY_DRINK_COUNT = 2
    ENERGY_REPLENISH = 40

    # Combo
    COMBO_TIMEOUT = 60 # steps
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_combo = pygame.font.SysFont("Verdana", 28, bold=True)
        
        self.render_mode = render_mode
        self._initialize_state_variables()
    
    def _initialize_state_variables(self):
        """Initializes all state variables to default values."""
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_energy = 0
        self.last_move_dir = pygame.Vector2(0, 1) # Default to down
        self.boost_timer = 0
        self.last_space_held = False
        
        self.drones = []
        self.drone_base_speed = self.DRONE_BASE_SPEED

        self.cameras = []
        
        self.resources = []
        self.particles = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.combo_timer = 0
        self.combo_multiplier = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_variables()

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_energy = self.PLAYER_MAX_ENERGY
        
        self.drone_base_speed = self.DRONE_BASE_SPEED
        self.drones = [self._spawn_drone() for _ in range(self.DRONE_INITIAL_COUNT)]
        
        self.cameras = [self._spawn_camera() for _ in range(self.CAMERA_COUNT)]
        
        self.resources = [self._spawn_resource() for _ in range(self.ENERGY_DRINK_COUNT)]
        
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 3))
            for _ in range(100)
        ]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.01 # Survival reward
        self.steps += 1
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        self._handle_input(movement, space_held)
        self._update_player()
        self._update_drones()
        self._update_cameras()
        self._update_particles()
        self._update_combo()

        # Spawn new entities
        if self.steps % self.DRONE_SPAWN_INTERVAL == 0 and len(self.drones) < self.DRONE_MAX_COUNT:
            self.drones.append(self._spawn_drone())
        if self.steps % self.DRONE_SPEED_INCREASE_INTERVAL == 0:
            self.drone_base_speed += self.DRONE_SPEED_INCREASE_RATE

        # Collisions and interactions
        reward += self._handle_collisions()
        
        self.player_energy -= self.PLAYER_ENERGY_DECAY
        self.score += 1 # Score is distance traveled (steps)

        terminated = False
        if self.player_energy <= 0:
            # sfx: player_power_down
            terminated = True
            reward = -10.0
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward = 50.0 # Survival bonus

        # Must be last check as it can set terminated = True
        if self._check_drone_collision():
            terminated = True
            reward = -10.0
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        
        if move_vec.length() > 0:
            self.last_move_dir = move_vec.normalize()
        
        if self.boost_timer <= 0:
            self.player_vel += move_vec * self.PLAYER_SPEED * 0.2 # 0.2 is acceleration factor
        
        # Boost activation on key press
        if space_held and not self.last_space_held and self.player_energy >= self.BOOST_COST:
            # sfx: player_boost
            self.player_energy -= self.BOOST_COST
            self.player_vel = self.last_move_dir * self.BOOST_SPEED
            self.boost_timer = self.BOOST_DURATION
            self._trigger_combo(2.0) # Reward for boosting
            self._add_particles(self.player_pos, 20, self.COLOR_PLAYER, 5)

        self.last_space_held = space_held

    def _update_player(self):
        if self.boost_timer > 0:
            self.boost_timer -= 1
        else:
            self.player_vel *= self.PLAYER_FRICTION
        
        self.player_pos += self.player_vel
        
        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_drones(self):
        is_alarm = any(c['alarm_timer'] > 0 for c in self.cameras)
        speed_multiplier = self.CAMERA_ALARM_SPEED_MULTIPLIER if is_alarm else 1.0
        
        for drone in self.drones:
            direction = (self.player_pos - drone['pos']).normalize()
            drone['pos'] += direction * self.drone_base_speed * speed_multiplier
    
    def _update_cameras(self):
        for camera in self.cameras:
            if camera['alarm_timer'] > 0:
                camera['alarm_timer'] -= 1
            dist_to_player = self.player_pos.distance_to(camera['pos'])
            if dist_to_player < self.CAMERA_RADIUS and camera['alarm_timer'] <= 0:
                # sfx: camera_alarm_trigger
                camera['alarm_timer'] = self.CAMERA_ALARM_DURATION

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_combo(self):
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo_multiplier = 1

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Resources
        for res in self.resources[:]:
            dist = self.player_pos.distance_to(res['pos'])
            if dist < self.PLAYER_SIZE + res['size']:
                # sfx: resource_collect
                self.player_energy = min(self.PLAYER_MAX_ENERGY, self.player_energy + self.ENERGY_REPLENISH)
                self.resources.remove(res)
                self.resources.append(self._spawn_resource())
                reward += 1.0
                self._add_particles(res['pos'], 15, self.COLOR_ENERGY, 3)
                reward += self._trigger_combo()
        return reward
    
    def _check_drone_collision(self):
        for drone in self.drones:
            if self.player_pos.distance_to(drone['pos']) < self.PLAYER_SIZE + self.DRONE_SIZE:
                # sfx: player_hit_explosion
                self._add_particles(self.player_pos, 50, self.COLOR_DRONE, 8)
                return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Parallax stars
        for x, y, speed in self.stars:
            # Move stars based on player velocity to create parallax
            star_pos_x = (x - self.player_pos.x * 0.1 * speed) % self.WIDTH
            star_pos_y = (y - self.player_pos.y * 0.1 * speed) % self.HEIGHT
            color_val = 50 * speed
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(star_pos_x), int(star_pos_y)), 1)
    
    def _render_game(self):
        # Cameras
        for cam in self.cameras:
            is_alarm = cam['alarm_timer'] > 0
            base_color = self.COLOR_DRONE if is_alarm else self.COLOR_CAMERA
            
            # Pulsing effect for alarm
            alpha = 100 + (math.sin(self.steps * 0.5) * 50) if is_alarm else 30
            pygame.gfxdraw.filled_circle(self.screen, int(cam['pos'].x), int(cam['pos'].y), self.CAMERA_RADIUS, (*base_color, alpha))
            pygame.gfxdraw.aacircle(self.screen, int(cam['pos'].x), int(cam['pos'].y), self.CAMERA_RADIUS, base_color)

        # Resources
        for res in self.resources:
            self._draw_glowing_circle(res['pos'], res['size'], res['color'])

        # Drones
        for drone in self.drones:
            self._draw_glowing_shape(drone['pos'], self.DRONE_SIZE, self.COLOR_DRONE, 4) # 4 sides for rhombus

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            p_color = (*p['color'], alpha)
            pygame.draw.circle(self.screen, p_color, (int(p['pos'].x), int(p['pos'].y)), int(p['life'] * 0.2))

        # Player
        player_color = self.COLOR_PLAYER
        if self.boost_timer > 0:
            # Brighter when boosting
            player_color = (200, 255, 255)
        self._draw_glowing_circle(self.player_pos, self.PLAYER_SIZE, player_color)
    
    def _render_ui(self):
        # Energy Bar
        bar_width = 200
        bar_height = 20
        energy_ratio = max(0, self.player_energy / self.PLAYER_MAX_ENERGY)
        current_bar_width = int(bar_width * energy_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, 10, current_bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 1)

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, 20))
        self.screen.blit(score_text, score_rect)

        # Combo Multiplier
        if self.combo_multiplier > 1:
            combo_text = self.font_combo.render(f"{self.combo_multiplier}x", True, self.COLOR_PLAYER)
            
            # Pulsing size effect
            scale = 1.0 + 0.1 * math.sin(self.combo_timer / self.COMBO_TIMEOUT * math.pi)
            scaled_font = pygame.font.SysFont("Verdana", int(28 * scale), bold=True)
            combo_text = scaled_font.render(f"{self.combo_multiplier}x", True, self.COLOR_PLAYER)

            combo_rect = combo_text.get_rect(topright=(self.WIDTH - 20, 15))
            self.screen.blit(combo_text, combo_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.player_energy,
            "combo": self.combo_multiplier,
        }

    def _spawn_entity(self, size):
        """Helper to spawn any entity away from the player."""
        while True:
            pos = pygame.Vector2(
                random.uniform(size, self.WIDTH - size),
                random.uniform(size, self.HEIGHT - size)
            )
            if self.player_pos.length() > 0 and pos.distance_to(self.player_pos) > 100:
                return pos
            elif self.player_pos.length() == 0: # Handle initial spawn
                return pos

    def _spawn_drone(self):
        # Spawn off-screen
        side = random.randint(0, 3)
        if side == 0: # Top
            pos = pygame.Vector2(random.uniform(0, self.WIDTH), -self.DRONE_SIZE)
        elif side == 1: # Bottom
            pos = pygame.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT + self.DRONE_SIZE)
        elif side == 2: # Left
            pos = pygame.Vector2(-self.DRONE_SIZE, random.uniform(0, self.HEIGHT))
        else: # Right
            pos = pygame.Vector2(self.WIDTH + self.DRONE_SIZE, random.uniform(0, self.HEIGHT))
        return {'pos': pos}

    def _spawn_camera(self):
        return {'pos': self._spawn_entity(self.CAMERA_RADIUS), 'alarm_timer': 0}

    def _spawn_resource(self):
        return {'pos': self._spawn_entity(10), 'size': 8, 'color': self.COLOR_ENERGY}

    def _add_particles(self, pos, count, color, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = random.randint(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'color': color, 'life': life, 'max_life': life})
    
    def _trigger_combo(self, base_reward=0.5):
        """Resets combo timer and increases multiplier. Returns reward for combo increase."""
        reward = 0
        if self.combo_timer > 0: # If continuing a combo
            self.combo_multiplier += 1
            reward = base_reward * self.combo_multiplier
            # sfx: combo_increase
        else: # Starting a new combo
            self.combo_multiplier = 2
        self.combo_timer = self.COMBO_TIMEOUT
        return reward

    def _draw_glowing_circle(self, pos, radius, color):
        """Draws a circle with a fake bloom effect."""
        x, y = int(pos.x), int(pos.y)
        for i in range(3, 0, -1):
            alpha = 80 // (i * i)
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius + i * 3, (*color, alpha))
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _draw_glowing_shape(self, pos, size, color, num_sides):
        """Draws a regular polygon with a fake bloom effect."""
        points = []
        for i in range(num_sides):
            angle = i * (2 * math.pi / num_sides) + (math.pi / 4) # Rotate for style
            x = pos.x + size * math.cos(angle)
            y = pos.y + size * math.sin(angle)
            points.append((int(x), int(y)))
        
        # Glow effect
        for i in range(3, 0, -1):
            alpha = 80 // (i * i)
            glow_points = []
            for p in points:
                vec = (pygame.Vector2(p) - pos).normalize() * i * 2
                glow_points.append(pos + vec)
            
            # This is a bit of a hack, we can't easily scale polygons. Draw lines instead.
            pygame.draw.lines(self.screen, (*color, alpha), True, glow_points, width=i*2)

        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you'll need to remove the headless environment variable setting
    # at the top of the file.
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Cyberpunk Runner")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not terminated:
        movement = 0 # No-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
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
            space = 1
            
        action = [movement, space, 0]
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()