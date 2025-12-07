import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:27:00.821352
# Source Brief: brief_01100.md
# Brief Index: 1100
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the player defends a central planet from incoming asteroids.
    The player controls a defender ship that orbits the planet and can fire two types of
    energy beams. The goal is to survive for a fixed duration.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Defend your home planet from a relentless asteroid shower. Pilot a powerful ship and unleash "
        "devastating energy beams to survive."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your ship. Press space to fire a standard beam and "
        "hold shift to unleash a powerful focused beam."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # --- Game Constants ---
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_STAR = (100, 100, 120)
        self.COLOR_PLANET = (40, 120, 60)
        self.COLOR_PLANET_DMG = (100, 40, 40)
        self.COLOR_DEFENDER = (255, 255, 255)
        self.COLOR_DEFENDER_GLOW = (200, 200, 255, 50)
        self.COLOR_ASTEROID = (180, 80, 80)
        self.COLOR_BEAM_STD = (100, 200, 255)
        self.COLOR_BEAM_FOCUS = (150, 255, 255)
        self.COLOR_EXPLOSION = (255, 200, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_ENERGY_BAR = (50, 150, 255)
        self.COLOR_ENERGY_EMPTY = (60, 60, 60)
        
        # Gameplay parameters
        self.MAX_STEPS = 1000
        self.PLANET_RADIUS = 50
        self.PLANET_POS = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        self.MAX_PLANET_HEALTH = 100
        
        self.DEFENDER_SPEED = 0.5
        self.DEFENDER_DRAG = 0.92
        self.DEFENDER_RADIUS = 10
        
        self.BEAM_ENERGY_MAX = 100
        self.BEAM_ENERGY_REGEN = 0.25
        self.BEAM_STD_COST = 15
        self.BEAM_STD_DMG = 35
        self.BEAM_STD_COOLDOWN = 10
        self.BEAM_FOCUS_COST = 40
        self.BEAM_FOCUS_DMG = 100
        self.BEAM_FOCUS_COOLDOWN = 25
        self.BEAM_LIFETIME = 8

        self.ASTEROID_SPAWN_RATE_INITIAL = 0.02 # 1 every 50 steps
        self.ASTEROID_SPAWN_RATE_INCREASE = 0.0001
        self.ASTEROID_SPAWN_RATE_MAX = 0.05
        self.ASTEROID_MAX_COUNT = 50
        self.FOCUSED_BEAM_UNLOCK_REQ = 10

        # Initialize state variables
        self.state_vars_initialized = False
        self.reset()

        # Validate implementation
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Planet state
        self.planet_health = self.MAX_PLANET_HEALTH
        
        # Defender state
        self.defender_pos = self.PLANET_POS + np.array([0, -self.PLANET_RADIUS - 30])
        self.defender_vel = np.array([0.0, 0.0])
        
        # Resource and cooldowns
        self.beam_energy = self.BEAM_ENERGY_MAX
        self.cooldown_std = 0
        self.cooldown_focus = 0
        
        # Progression
        self.asteroids_destroyed = 0
        self.focused_beam_unlocked = False
        self.unlock_notification_timer = 0
        
        # Dynamic lists
        self.asteroids = []
        self.beams = []
        self.particles = []
        
        # World state
        self.asteroid_spawn_rate = self.ASTEROID_SPAWN_RATE_INITIAL
        self._generate_starfield()

        self.state_vars_initialized = True
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if not self.state_vars_initialized:
            # This is a fallback, but reset should always be called first.
            self.reset()
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # 1. Handle Input & Actions
        self._handle_input(action)
        
        # 2. Update Game Entities
        self._update_defender()
        self._update_beams()
        self._update_asteroids()
        self._update_particles()
        
        # 3. Handle Collisions and Game Events
        destroyed_count, planet_hits = self._handle_collisions()
        self.asteroids_destroyed += destroyed_count
        self.planet_health -= planet_hits * 25 # Each hit takes 25 health
        
        # 4. Spawn new entities
        self._spawn_asteroids()
        
        # 5. Update game state and check for progression
        self._update_game_state()
        
        # 6. Calculate Reward
        reward += 0.01  # Small reward for surviving a step
        reward += destroyed_count * 1.0 # Reward for destroying asteroids
        
        # 7. Check for Termination
        terminated = False
        if self.planet_health <= 0:
            self.game_over = True
            terminated = True
            reward = -100.0 # Large penalty for losing
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            reward = 100.0 # Large reward for winning
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        accel = np.array([0.0, 0.0])
        if movement == 1: accel[1] -= self.DEFENDER_SPEED # Up
        if movement == 2: accel[1] += self.DEFENDER_SPEED # Down
        if movement == 3: accel[0] -= self.DEFENDER_SPEED # Left
        if movement == 4: accel[0] += self.DEFENDER_SPEED # Right
        self.defender_vel += accel
        
        # Firing
        if self.cooldown_std > 0: self.cooldown_std -= 1
        if self.cooldown_focus > 0: self.cooldown_focus -= 1
        
        # Aim direction is always from defender to planet center
        aim_vec = (self.PLANET_POS - self.defender_pos)
        dist = np.linalg.norm(aim_vec)
        aim_dir = aim_vec / dist if dist > 0 else np.array([0, 1])

        # Fire Standard Beam
        if space_held and self.cooldown_std == 0 and self.beam_energy >= self.BEAM_STD_COST:
            self.beam_energy -= self.BEAM_STD_COST
            self.cooldown_std = self.BEAM_STD_COOLDOWN
            self.beams.append({
                "start": self.defender_pos.copy(),
                "end": self.defender_pos + aim_dir * self.WIDTH,
                "type": "std", "lifetime": self.BEAM_LIFETIME
            })
            
        # Fire Focused Beam
        if shift_held and self.focused_beam_unlocked and self.cooldown_focus == 0 and self.beam_energy >= self.BEAM_FOCUS_COST:
            self.beam_energy -= self.BEAM_FOCUS_COST
            self.cooldown_focus = self.BEAM_FOCUS_COOLDOWN
            self.beams.append({
                "start": self.defender_pos.copy(),
                "end": self.defender_pos + aim_dir * self.WIDTH,
                "type": "focus", "lifetime": self.BEAM_LIFETIME
            })

    def _update_defender(self):
        self.defender_vel *= self.DEFENDER_DRAG
        self.defender_pos += self.defender_vel
        # Clamp position to screen bounds
        self.defender_pos[0] = np.clip(self.defender_pos[0], self.DEFENDER_RADIUS, self.WIDTH - self.DEFENDER_RADIUS)
        self.defender_pos[1] = np.clip(self.defender_pos[1], self.DEFENDER_RADIUS, self.HEIGHT - self.DEFENDER_RADIUS)

    def _update_beams(self):
        for beam in self.beams:
            beam['lifetime'] -= 1
        self.beams = [b for b in self.beams if b['lifetime'] > 0]

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']
        # Remove asteroids that are far off-screen
        self.asteroids = [a for a in self.asteroids if -100 < a['pos'][0] < self.WIDTH + 100 and -100 < a['pos'][1] < self.HEIGHT + 100]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_game_state(self):
        self.steps += 1
        self.score = self.asteroids_destroyed * 10
        self.beam_energy = min(self.BEAM_ENERGY_MAX, self.beam_energy + self.BEAM_ENERGY_REGEN)
        self.asteroid_spawn_rate = min(self.ASTEROID_SPAWN_RATE_MAX, self.asteroid_spawn_rate + self.ASTEROID_SPAWN_RATE_INCREASE)

        if not self.focused_beam_unlocked and self.asteroids_destroyed >= self.FOCUSED_BEAM_UNLOCK_REQ:
            self.focused_beam_unlocked = True
            self.unlock_notification_timer = 120 # Show for 4 seconds at 30fps
        
        if self.unlock_notification_timer > 0:
            self.unlock_notification_timer -= 1

    def _spawn_asteroids(self):
        if len(self.asteroids) < self.ASTEROID_MAX_COUNT and self.np_random.random() < self.asteroid_spawn_rate:
            # Spawn on an edge
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = np.array([self.np_random.uniform(0, self.WIDTH), -50.0])
            elif edge == 1: # Bottom
                pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 50.0])
            elif edge == 2: # Left
                pos = np.array([-50.0, self.np_random.uniform(0, self.HEIGHT)])
            else: # Right
                pos = np.array([self.WIDTH + 50.0, self.np_random.uniform(0, self.HEIGHT)])
            
            # Aim towards the planet with some variance
            target_pos = self.PLANET_POS + self.np_random.uniform(-self.PLANET_RADIUS, self.PLANET_RADIUS, 2)
            direction = (target_pos - pos)
            norm = np.linalg.norm(direction)
            vel = (direction / norm if norm > 0 else np.array([0,0])) * self.np_random.uniform(1.0, 2.5)
            
            radius = self.np_random.uniform(10, 25)
            self.asteroids.append({
                "pos": pos, "vel": vel, "radius": radius,
                "health": radius * 4, "angle": 0,
                "rot_speed": self.np_random.uniform(-0.05, 0.05)
            })

    def _handle_collisions(self):
        destroyed_count = 0
        planet_hits = 0
        
        # Asteroid-beam collisions
        for beam in self.beams:
            damage = self.BEAM_STD_DMG if beam['type'] == 'std' else self.BEAM_FOCUS_DMG
            for asteroid in self.asteroids:
                if self._line_segment_circle_collision(beam['start'], beam['end'], asteroid['pos'], asteroid['radius']):
                    asteroid['health'] -= damage
                    self._create_particles(asteroid['pos'], 10, self.COLOR_BEAM_STD, 2)
        
        # Asteroid destruction
        remaining_asteroids = []
        for asteroid in self.asteroids:
            if asteroid['health'] <= 0:
                destroyed_count += 1
                self._create_explosion(asteroid['pos'], int(asteroid['radius']))
            else:
                remaining_asteroids.append(asteroid)
        self.asteroids = remaining_asteroids
        
        # Asteroid-planet collisions
        remaining_asteroids = []
        for asteroid in self.asteroids:
            dist = np.linalg.norm(asteroid['pos'] - self.PLANET_POS)
            if dist < asteroid['radius'] + self.PLANET_RADIUS:
                planet_hits += 1
                self._create_explosion(asteroid['pos'], int(asteroid['radius']))
            else:
                remaining_asteroids.append(asteroid)
        self.asteroids = remaining_asteroids
        
        return destroyed_count, planet_hits

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_planet()
        self._render_asteroids()
        self._render_beams()
        self._render_particles()
        self._render_defender()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return { "score": self.score, "steps": self.steps, "planet_health": self.planet_health }

    # --- Rendering Methods ---

    def _render_starfield(self):
        for star in self.starfield:
            pygame.gfxdraw.pixel(self.screen, int(star[0]), int(star[1]), self.COLOR_STAR)

    def _render_planet(self):
        pos = (int(self.PLANET_POS[0]), int(self.PLANET_POS[1]))
        rad = int(self.PLANET_RADIUS)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad, self.COLOR_PLANET)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], rad, self.COLOR_PLANET)
        
        # Render damage
        damage_level = (self.MAX_PLANET_HEALTH - self.planet_health) / self.MAX_PLANET_HEALTH
        if damage_level > 0.2:
            pygame.draw.line(self.screen, self.COLOR_PLANET_DMG, (pos[0]-10, pos[1]-20), (pos[0]+15, pos[1]+5), 2)
        if damage_level > 0.5:
            pygame.draw.line(self.screen, self.COLOR_PLANET_DMG, (pos[0]+25, pos[1]-15), (pos[0]-5, pos[1]+22), 2)
        if damage_level > 0.8:
            pygame.gfxdraw.filled_circle(self.screen, pos[0]+20, pos[1]+20, 7, self.COLOR_PLANET_DMG)

    def _render_defender(self):
        pos = self.defender_pos
        rad = self.DEFENDER_RADIUS
        
        # Glow effect
        glow_surf = pygame.Surface((rad * 4, rad * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_DEFENDER_GLOW, (rad * 2, rad * 2), rad * 1.5)
        self.screen.blit(glow_surf, (int(pos[0] - rad * 2), int(pos[1] - rad * 2)))
        
        # Main body
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), rad, self.COLOR_DEFENDER)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), rad, self.COLOR_DEFENDER)
        
    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = (int(asteroid['pos'][0]), int(asteroid['pos'][1]))
            rad = int(asteroid['radius'])
            # Simple square asteroid
            points = [
                (-rad, -rad), (rad, -rad), (rad, rad), (-rad, rad)
            ]
            # Rotate points
            angle = asteroid['angle']
            rotated_points = []
            for p in points:
                x = p[0] * math.cos(angle) - p[1] * math.sin(angle) + pos[0]
                y = p[0] * math.sin(angle) + p[1] * math.cos(angle) + pos[1]
                rotated_points.append((x, y))
            
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_ASTEROID)

    def _render_beams(self):
        for beam in self.beams:
            start = (int(beam['start'][0]), int(beam['start'][1]))
            end = (int(beam['end'][0]), int(beam['end'][1]))
            if beam['type'] == 'std':
                pygame.draw.line(self.screen, self.COLOR_BEAM_STD, start, end, 2)
            else: # Focus beam
                pygame.draw.line(self.screen, self.COLOR_BEAM_FOCUS, start, end, 5)
                pygame.draw.line(self.screen, (255,255,255), start, end, 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                # Using a rect for particles is faster than circles
                rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
                pygame.draw.rect(self.screen, color, rect)

    def _render_ui(self):
        # Score and Time
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        time_text = self.font_small.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Planet Health Bar
        health_percent = self.planet_health / self.MAX_PLANET_HEALTH
        bar_width = 100
        bar_height = 10
        bar_x = self.WIDTH // 2 - bar_width // 2
        bar_y = self.HEIGHT - 20
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_EMPTY, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PLANET, (bar_x, bar_y, int(bar_width * health_percent), bar_height))

        # Beam Energy Bar
        energy_percent = self.beam_energy / self.BEAM_ENERGY_MAX
        bar_width = 150
        bar_x = 10
        bar_y = self.HEIGHT - 20
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_EMPTY, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (bar_x, bar_y, int(bar_width * energy_percent), bar_height))

        # Unlock Notification
        if self.unlock_notification_timer > 0:
            alpha = min(255, int(255 * (self.unlock_notification_timer / 30.0)))
            text = self.font_large.render("FOCUSED BEAM UNLOCKED [SHIFT]", True, self.COLOR_BEAM_FOCUS)
            text.set_alpha(alpha)
            text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 100))
            self.screen.blit(text, text_rect)

    # --- Helper Methods ---

    def _generate_starfield(self):
        self.starfield = []
        for _ in range(150):
            self.starfield.append((self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)))

    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': life, 'max_life': life,
                'color': color, 'size': self.np_random.uniform(2, 5)
            })

    def _create_explosion(self, pos, radius):
        self._create_particles(pos, radius * 2, self.COLOR_EXPLOSION, 1.5)

    def _line_segment_circle_collision(self, start, end, circle_center, circle_radius):
        p1 = start
        p2 = end
        p3 = circle_center
        
        # Vector from start to end of line segment
        d = p2 - p1
        # Vector from start to circle center
        f = p1 - p3
        
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - circle_radius**2
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return False
        else:
            discriminant = math.sqrt(discriminant)
            t1 = (-b - discriminant) / (2*a)
            t2 = (-b + discriminant) / (2*a)
            
            # Check if intersection is within the line segment
            if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                return True
            # Check if segment is fully inside circle
            if t1 < 0 and t2 > 1:
                return True
        return False

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display window
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Planet Defender")
    
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Rendering to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for playability

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()