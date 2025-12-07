import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:20:14.754212
# Source Brief: brief_02249.md
# Brief Index: 2249
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Entities ---

class Particle:
    """A single particle for effects like explosions."""
    def __init__(self, x, y, color, size, speed, angle, gravity, decay):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(speed, 0).rotate(angle)
        self.color = color
        self.size = size
        self.gravity = gravity
        self.decay = decay

    def update(self):
        self.pos += self.vel
        self.vel.y += self.gravity
        self.size -= self.decay
        return self.size > 0

    def draw(self, surface):
        if self.size > 0:
            pygame.draw.circle(surface, self.color, self.pos, int(self.size))

class Projectile:
    """An energy blast fired by a robot."""
    def __init__(self, x, y, angle, damage, speed, color):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(speed, 0).rotate_rad(angle)
        self.damage = damage
        self.color = color
        self.radius = 4

    def update(self):
        self.pos += self.vel

    def draw(self, surface):
        end_pos = self.pos - self.vel.normalize() * 15
        pygame.draw.line(surface, self.color, self.pos, end_pos, self.radius)
        # Glow effect
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius + 2, (*self.color, 60))

class Alien:
    """An invader entity."""
    def __init__(self, x, y, wave_number):
        self.pos = pygame.Vector2(x, y)
        self.health = 1 + wave_number // 5
        self.max_health = self.health
        self.radius = 10 + wave_number // 10
        self.base_speed = 0.8 + (wave_number * 0.05)
        self.color = (255, 50, 50)
        
        # Movement pattern
        self.pattern_type = random.choice(['straight', 'sine', 'diagonal'])
        self.sine_freq = random.uniform(0.01, 0.03)
        self.sine_amp = random.uniform(20, 50)
        self.sine_phase = random.uniform(0, 2 * math.pi)
        self.start_x = x

    def update(self, robots):
        target_robot = min(robots, key=lambda r: self.pos.distance_to(r.pos), default=None)
        if target_robot:
            direction = (target_robot.pos - self.pos).normalize()
        else:
            direction = pygame.Vector2(0, 1) # Head down if no robots left
        
        if self.pattern_type == 'straight':
            self.pos += direction * self.base_speed
        elif self.pattern_type == 'sine':
            self.pos.y += self.base_speed
            offset = math.sin(self.pos.y * self.sine_freq + self.sine_phase) * self.sine_amp
            self.pos.x = self.start_x + offset
        elif self.pattern_type == 'diagonal':
            self.pos += direction * self.base_speed
            self.pos.x += math.sin(self.pos.y * 0.02) * 1.5
            
        # Keep within horizontal bounds
        self.pos.x = max(self.radius, min(640 - self.radius, self.pos.x))

    def draw(self, surface):
        # Body
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, (255, 150, 150))
        # Health bar
        if self.health < self.max_health:
            bar_width = self.radius * 2
            bar_height = 4
            health_pct = self.health / self.max_health
            pygame.draw.rect(surface, (50, 50, 50), (self.pos.x - bar_width/2, self.pos.y - self.radius - 8, bar_width, bar_height))
            pygame.draw.rect(surface, (0, 255, 0), (self.pos.x - bar_width/2, self.pos.y - self.radius - 8, bar_width * health_pct, bar_height))

class Robot:
    """A player-controlled robot defender."""
    def __init__(self, x, y, robot_type):
        self.pos = pygame.Vector2(x, y)
        self.type = robot_type
        self.health = 100
        self.max_health = 100
        self.aim_angle = -math.pi / 2  # Start aiming up
        self.aim_speed = 0.08
        self.is_active = False
        self.radius = 15

        if self.type == "Balanced":
            self.color = (50, 150, 255)
            self.max_energy = 100
            self.energy_regen = 0.3
            self.fire_cost = 20
            self.fire_cooldown_max = 15 # steps
            self.proj_damage = 1
        elif self.type == "Rapid":
            self.color = (100, 255, 100)
            self.max_energy = 80
            self.energy_regen = 0.5
            self.fire_cost = 10
            self.fire_cooldown_max = 7
            self.proj_damage = 0.7
        elif self.type == "Heavy":
            self.color = (255, 150, 50)
            self.max_energy = 150
            self.energy_regen = 0.2
            self.fire_cost = 40
            self.fire_cooldown_max = 30
            self.proj_damage = 3
        
        self.energy = self.max_energy
        self.fire_cooldown = 0

    def update(self):
        self.energy = min(self.max_energy, self.energy + self.energy_regen)
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0

    def draw(self, surface):
        # Body
        poly_points = [
            self.pos + pygame.Vector2(0, -self.radius).rotate_rad(self.aim_angle),
            self.pos + pygame.Vector2(-self.radius*0.8, self.radius*0.8).rotate_rad(self.aim_angle),
            self.pos + pygame.Vector2(self.radius*0.8, self.radius*0.8).rotate_rad(self.aim_angle),
        ]
        int_points = [(int(p.x), int(p.y)) for p in poly_points]
        
        glow_color = (*self.color, 20) if self.is_active else (*self.color, 10)
        body_color = self.color if self.is_active else tuple(c // 2 for c in self.color)
        
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius + 5, glow_color)
        pygame.gfxdraw.filled_polygon(surface, int_points, body_color)
        pygame.gfxdraw.aapolygon(surface, int_points, body_color)
        
        # Aiming line for active robot
        if self.is_active:
            aim_end = self.pos + pygame.Vector2(60, 0).rotate_rad(self.aim_angle)
            pygame.draw.line(surface, (255, 255, 255, 100), self.pos, aim_end, 1)

# --- Main Gymnasium Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend the city from waves of alien invaders by controlling a squad of powerful robots. "
        "Aim and fire to destroy aliens before they breach your defenses."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim your robot's weapon. "
        "Press space to fire. Press shift to switch between available robots."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and timing
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_CITY = (40, 40, 50)
        self.COLOR_UI = (200, 200, 220)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.wave_cooldown = 0
        self.robots = []
        self.aliens = []
        self.projectiles = []
        self.particles = []
        self.city_buildings = []
        self.stars = []
        self.active_robot_idx = 0
        self.last_shift_press = False
        self.unlocked_robots = set()

        self.reset()
        # self.validate_implementation() # Optional validation call
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset core state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.wave_cooldown = self.FPS * 3 # 3 seconds before first wave
        
        # Reset entities
        self.aliens.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        # Reset robots and unlocks
        self.robots = [Robot(self.WIDTH / 2, self.HEIGHT - 30, "Balanced")]
        self.unlocked_robots = {"Balanced"}
        self.active_robot_idx = 0
        self.robots[0].is_active = True
        
        # Reset input state
        self.last_shift_press = False
        
        # Procedurally generate background
        self._generate_background()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        active_robot = self.robots[self.active_robot_idx]

        # Aiming
        if movement == 1: active_robot.aim_angle -= active_robot.aim_speed # Up
        if movement == 2: active_robot.aim_angle += active_robot.aim_speed # Down
        if movement == 3: active_robot.aim_angle -= active_robot.aim_speed # Left
        if movement == 4: active_robot.aim_angle += active_robot.aim_speed # Right
        # Clamp angle to prevent aiming downwards
        active_robot.aim_angle = max(-math.pi, min(0, active_robot.aim_angle))

        # Firing
        if space_held and active_robot.fire_cooldown == 0 and active_robot.energy >= active_robot.fire_cost:
            # SFX: Laser fire
            active_robot.energy -= active_robot.fire_cost
            active_robot.fire_cooldown = active_robot.fire_cooldown_max
            self.projectiles.append(Projectile(
                active_robot.pos.x, active_robot.pos.y, active_robot.aim_angle,
                active_robot.proj_damage, 10, (100, 255, 100)
            ))
            reward -= 0.01 # Cost for firing

        # Cycle active robot (on press, not hold)
        if shift_held and not self.last_shift_press and len(self.robots) > 1:
            active_robot.is_active = False
            self.active_robot_idx = (self.active_robot_idx + 1) % len(self.robots)
            self.robots[self.active_robot_idx].is_active = True
        self.last_shift_press = shift_held

        # --- Update Game Logic ---
        for r in self.robots: r.update()
        for a in self.aliens: a.update(self.robots)
        for p in self.projectiles: p.update()
        self.particles = [p for p in self.particles if p.update()]
        
        # --- Handle Collisions and Removals ---
        reward += self._handle_collisions()
        
        # Remove off-screen projectiles
        self.projectiles = [p for p in self.projectiles if 0 < p.pos.x < self.WIDTH and 0 < p.pos.y < self.HEIGHT]
        
        # Alien reaches city
        aliens_to_remove = []
        for alien in self.aliens:
            if alien.pos.y > self.HEIGHT - 50: # Breach zone
                aliens_to_remove.append(alien)
                # Damage the closest robot
                if self.robots:
                    closest_robot = min(self.robots, key=lambda r: alien.pos.distance_to(r.pos))
                    if closest_robot.take_damage(25): # SFX: Robot damage
                        self._create_explosion(closest_robot.pos, 50)
                        self.robots = [r for r in self.robots if r.health > 0]
                        if not self.robots:
                            self.game_over = True
                        else:
                            # If active robot was destroyed, switch to another
                            self.active_robot_idx = min(self.active_robot_idx, len(self.robots) - 1)
                            self.robots[self.active_robot_idx].is_active = True

        if aliens_to_remove:
            self.aliens = [a for a in self.aliens if a not in aliens_to_remove]
            # SFX: City breach alarm

        # --- Wave Management ---
        if not self.aliens and not self.game_over:
            if self.wave_cooldown > 0:
                self.wave_cooldown -= 1
            else:
                self.wave_number += 1
                reward += 10 # Wave clear bonus
                self._spawn_wave()
                self._check_unlocks()
                self.wave_cooldown = self.FPS * 3 # 3 seconds between waves

        # --- Termination Conditions ---
        if not self.robots:
            self.game_over = True
            reward = -100 # Game over penalty
        
        if self.steps >= 10000:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )
    
    def _handle_collisions(self):
        reward = 0
        projectiles_to_remove = set()
        aliens_to_remove = set()

        for i, p in enumerate(self.projectiles):
            for j, a in enumerate(self.aliens):
                if p.pos.distance_to(a.pos) < a.radius:
                    # SFX: Alien hit
                    a.health -= p.damage
                    projectiles_to_remove.add(i)
                    reward += 0.1 # Hit bonus
                    if a.health <= 0:
                        # SFX: Alien explosion
                        aliens_to_remove.add(j)
                        self._create_explosion(a.pos, 30)
                        self.score += 10 * (1 + self.wave_number * 0.1)
                    else:
                        self._create_explosion(p.pos, 5) # Small hit spark
                    break 

        if projectiles_to_remove or aliens_to_remove:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
            self.aliens = [a for j, a in enumerate(self.aliens) if j not in aliens_to_remove]

        return reward

    def _spawn_wave(self):
        num_aliens = 3 + self.wave_number // 2
        for _ in range(num_aliens):
            x = random.uniform(50, self.WIDTH - 50)
            y = random.uniform(-100, -20)
            self.aliens.append(Alien(x, y, self.wave_number))

    def _check_unlocks(self):
        if self.wave_number >= 5 and "Rapid" not in self.unlocked_robots:
            self.unlocked_robots.add("Rapid")
            self.robots.append(Robot(self.WIDTH * 0.25, self.HEIGHT - 30, "Rapid"))
        if self.wave_number >= 10 and "Heavy" not in self.unlocked_robots:
            self.unlocked_robots.add("Heavy")
            self.robots.append(Robot(self.WIDTH * 0.75, self.HEIGHT - 30, "Heavy"))
        
        # Ensure health and energy is full for all robots at start of wave
        for r in self.robots:
            r.health = r.max_health
            r.energy = r.max_energy

    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            color = random.choice([(255, 255, 255), (255, 200, 0), (255, 100, 0)])
            size = random.uniform(2, 6)
            speed = random.uniform(1, 5)
            angle = random.uniform(0, 360)
            gravity = 0.1
            decay = 0.1
            self.particles.append(Particle(pos.x, pos.y, color, size, speed, angle, gravity, decay))

    def _generate_background(self):
        # Starfield
        self.stars = []
        for _ in range(100):
            self.stars.append({
                "pos": pygame.Vector2(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)),
                "speed": random.uniform(0.1, 0.5),
                "size": random.randint(1, 2)
            })
        
        # Cityscape
        self.city_buildings = []
        x = -20
        while x < self.WIDTH:
            w = random.randint(30, 80)
            h = random.randint(50, self.HEIGHT // 2)
            self.city_buildings.append(pygame.Rect(x, self.HEIGHT - h, w, h))
            x += w - random.randint(5, 15)

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "robots_left": len(self.robots),
        }

    def _render_all(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        
        # Game elements
        for p in self.projectiles: p.draw(self.screen)
        for r in self.robots: r.draw(self.screen)
        for a in self.aliens: a.draw(self.screen)
        for p in self.particles: p.draw(self.screen)
        
        # UI
        self._render_ui()

        if self.game_over:
            self._render_game_over()

    def _render_background(self):
        # Stars
        for star in self.stars:
            star["pos"].y += star["speed"]
            if star["pos"].y > self.HEIGHT:
                star["pos"].y = 0
                star["pos"].x = random.randint(0, self.WIDTH)
            pygame.draw.circle(self.screen, (200, 200, 220), star["pos"], star["size"] / 2)
        
        # City
        for building in self.city_buildings:
            pygame.draw.rect(self.screen, self.COLOR_CITY, building)
            # Windows
            for _ in range(int(building.width * building.height / 500)):
                win_x = building.x + random.randint(5, building.width - 5)
                win_y = building.y + random.randint(5, building.height - 5)
                if random.random() < 0.3:
                    pygame.draw.rect(self.screen, (255, 255, 100), (win_x, win_y, 2, 2))

    def _render_ui(self):
        # Score and Wave
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(wave_text, (10, 35))

        # Robot status
        y_offset = 10
        for i, robot in enumerate(self.robots):
            # Health Bar
            health_pct = max(0, robot.health / robot.max_health)
            pygame.draw.rect(self.screen, (50,0,0), (self.WIDTH - 110, y_offset, 100, 10))
            pygame.draw.rect(self.screen, (0,200,0), (self.WIDTH - 110, y_offset, 100 * health_pct, 10))
            
            # Energy Bar
            energy_pct = max(0, robot.energy / robot.max_energy)
            pygame.draw.rect(self.screen, (0,0,50), (self.WIDTH - 110, y_offset + 12, 100, 10))
            pygame.draw.rect(self.screen, (0,100,200), (self.WIDTH - 110, y_offset + 12, 100 * energy_pct, 10))

            # Robot Type and Active Indicator
            type_color = self.COLOR_UI if robot.is_active else (100, 100, 120)
            type_text = self.font_small.render(robot.type, True, type_color)
            self.screen.blit(type_text, (self.WIDTH - 110 - type_text.get_width() - 5, y_offset + 5))

            y_offset += 35
        
        # Wave transition text
        if not self.aliens and self.wave_cooldown > 0 and not self.game_over:
            secs_left = math.ceil(self.wave_cooldown / self.FPS)
            text = f"WAVE {self.wave_number + 1} IN {secs_left}"
            if self.wave_number == 0:
                text = f"FIRST WAVE IN {secs_left}"
            
            wave_alert_text = self.font_large.render(text, True, self.COLOR_UI)
            pos = wave_alert_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(wave_alert_text, pos)

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0, 180))
        self.screen.blit(s, (0,0))
        
        game_over_text = self.font_large.render("CITY HAS FALLEN", True, (255, 50, 50))
        pos = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
        self.screen.blit(game_over_text, pos)
        
        final_score_text = self.font_small.render(f"FINAL SCORE: {int(self.score)}", True, self.COLOR_UI)
        pos = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
        self.screen.blit(final_score_text, pos)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == "__main__":
    # The main loop now requires a visible display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Robot City Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Action state
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = 0
    shift_held = 0
    
    running = True
    while running:
        # --- Pygame event handling for manual control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            
            # Action mapping
            # Note: Up/Down and Left/Right are mapped to the same angle change
            # for simplicity in manual play, but they are distinct actions.
            movement = 0
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Render the observation to the display window ---
        # The observation is (H, W, C), but pygame needs (W, H) surface.
        # We need to transpose it back.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Limit FPS for playability
        env.clock.tick(env.FPS)

    env.close()