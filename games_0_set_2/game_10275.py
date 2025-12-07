import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game entities
class Particle:
    def __init__(self, pos, vel, size, color, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.size = size
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            color_with_alpha = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.size), color_with_alpha)

class Projectile:
    def __init__(self, pos, vel, color, damage, size=3):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.damage = damage
        self.size = size
        self.lifetime = 120  # 4 seconds at 30 FPS

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1

    def draw(self, surface):
        end_pos = self.pos - self.vel.normalize() * 10
        pygame.draw.line(surface, self.color, (int(self.pos.x), int(self.pos.y)), (int(end_pos.x), int(end_pos.y)), self.size)
        # Glow effect
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.size + 2, self.color + (50,))


class Ship:
    def __init__(self, pos, ship_type, color, team):
        self.pos = pygame.Vector2(pos)
        self.ship_type = ship_type
        self.color = color
        self.team = team # 'player' or 'enemy'
        self.vel = pygame.Vector2(0, 0)
        self.angle = -90 if team == 'player' else 90
        self.target = None
        self.cooldown = 0
        self.flash_timer = 0
        self.is_combo_participant = False

        # Type-specific stats
        if self.ship_type == 'fighter':
            self.max_health = 100
            self.speed = 1.5
            self.turn_speed = 3
            self.fire_rate = 30
            self.proj_speed = 8
            self.proj_damage = 10
        elif self.ship_type == 'interceptor':
            self.max_health = 70
            self.speed = 2.5
            self.turn_speed = 5
            self.fire_rate = 15
            self.proj_speed = 12
            self.proj_damage = 6
        elif self.ship_type == 'bomber':
            self.max_health = 150
            self.speed = 1.0
            self.turn_speed = 2
            self.fire_rate = 60
            self.proj_speed = 5
            self.proj_damage = 25
        
        self.health = self.max_health

    def update(self, all_enemies, all_players):
        self.cooldown = max(0, self.cooldown - 1)
        self.flash_timer = max(0, self.flash_timer - 1)
        
        # Target acquisition
        targets = all_enemies if self.team == 'player' else all_players
        if not self.target or self.target.health <= 0 or random.random() < 0.01:
            self.target = self._find_closest_target(targets)

        # Movement and Rotation
        if self.target:
            target_vec = self.target.pos - self.pos
            target_dist = target_vec.length()
            target_angle = target_vec.angle_to(pygame.Vector2(1, 0))

            # Turn towards target
            angle_diff = (target_angle - self.angle + 180) % 360 - 180
            self.angle += np.clip(angle_diff, -self.turn_speed, self.turn_speed)

            # Move
            if target_dist > 200: # Move closer if too far
                self.vel = pygame.Vector2(1, 0).rotate(self.angle) * self.speed
            elif target_dist < 150: # Back away if too close
                self.vel = pygame.Vector2(1, 0).rotate(self.angle) * -self.speed * 0.5
            else: # Strafe
                 self.vel = pygame.Vector2(0, 1).rotate(self.angle) * self.speed * 0.7
        else:
            self.vel *= 0.9 # Drift to a stop

        self.pos += self.vel
        self.pos.x = np.clip(self.pos.x, 20, 620)
        self.pos.y = np.clip(self.pos.y, 20, 340)

    def _find_closest_target(self, targets):
        closest_target = None
        min_dist = float('inf')
        for t in targets:
            dist = self.pos.distance_to(t.pos)
            if dist < min_dist:
                min_dist = dist
                closest_target = t
        return closest_target

    def take_damage(self, amount):
        self.health -= amount
        self.flash_timer = 5
        return self.health <= 0

    def draw(self, surface):
        # Base shape
        points = [
            pygame.Vector2(15, 0),
            pygame.Vector2(-10, 10),
            pygame.Vector2(-5, 0),
            pygame.Vector2(-10, -10)
        ]
        rotated_points = [p.rotate(self.angle) + self.pos for p in points]
        
        # Main color or flash color
        draw_color = (255, 255, 255) if self.flash_timer > 0 else self.color
        
        # Draw with antialiasing
        int_points = [(int(p.x), int(p.y)) for p in rotated_points]
        pygame.gfxdraw.aapolygon(surface, int_points, draw_color)
        pygame.gfxdraw.filled_polygon(surface, int_points, draw_color)

        # Glow effect
        glow_color = self.color + (100,) if not self.is_combo_participant else ((255, 255, 0) + (150,))
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), 18, glow_color)
        
        # Health bar
        if self.health < self.max_health:
            bar_width = 20
            bar_height = 3
            health_pct = self.health / self.max_health
            health_bar_pos = self.pos + pygame.Vector2(-bar_width / 2, -20)
            pygame.draw.rect(surface, (50, 50, 50), (health_bar_pos.x, health_bar_pos.y, bar_width, bar_height))
            pygame.draw.rect(surface, (0, 255, 0), (health_bar_pos.x, health_bar_pos.y, bar_width * health_pct, bar_height))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Deploy and command a fleet of starships to defeat waves of enemies. "
        "Position your ships strategically to form powerful combos."
    )
    user_guide = (
        "Use arrow keys to move the deployment cursor. Press space to deploy the selected ship "
        "or activate a combo. Press shift to cycle through available ship types."
    )
    auto_advance = False

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    UI_HEIGHT = 60
    GAME_HEIGHT = HEIGHT - UI_HEIGHT
    
    COLOR_BG = (10, 15, 30)
    COLOR_UI_BG = (20, 30, 50)
    COLOR_UI_FRAME = (100, 120, 150)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (0, 255, 128)
    COLOR_ENEMY = (120, 40, 60)
    
    SHIP_TYPES = {
        'fighter': {'color': (255, 50, 50), 'cost': 1},
        'interceptor': {'color': (50, 255, 50), 'cost': 2},
        'bomber': {'color': (50, 150, 255), 'cost': 3},
    }
    
    MAX_STEPS = 5000
    MAX_WAVES = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables are initialized in reset()
        self.stars = []
        self.player_ships = []
        self.enemy_ships = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.GAME_HEIGHT / 2)
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False
        self.available_ship_types = []
        self.selected_ship_type_idx = 0
        self.prev_space_held = 0
        self.prev_shift_held = 0
        self.armed_combo = None
        
        self._generate_stars()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_ships = []
        self.enemy_ships = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.GAME_HEIGHT / 2)
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False
        self.armed_combo = None

        self.available_ship_types = ['fighter']
        self.selected_ship_type_idx = 0
        
        self.prev_space_held = 0
        self.prev_shift_held = 0

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self._handle_input(movement, space_pressed, shift_pressed)
        
        # --- 2. Update Game Logic ---
        # Update ships
        for ship in self.player_ships + self.enemy_ships:
            ship.update(self.enemy_ships, self.player_ships)
            # Firing logic
            if ship.cooldown == 0 and ship.target and ship.pos.distance_to(ship.target.pos) < 300:
                reward += self._fire_ship(ship)
        
        # Update projectiles
        for p in self.player_projectiles + self.enemy_projectiles:
            p.update()
        
        # Update particles
        for p in self.particles:
            p.update()

        # --- 3. Handle Collisions & Damage ---
        reward += self._handle_collisions()

        # --- 4. Clean up dead entities ---
        self._cleanup_entities()

        # --- 5. Check Game State ---
        # Wave completion
        if not self.enemy_ships and not self.game_over:
            reward += 100
            if self.wave >= self.MAX_WAVES:
                self.game_over = True # VICTORY
            else:
                self._start_next_wave()

        # Defeat condition
        if not self.player_ships and self.wave > 0:
            if not self.game_over: # only apply penalty once
                reward -= 100
            self.game_over = True

        # Check for armed combos
        self._check_combos()

        # --- 6. Finalize Step ---
        self.score += reward
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Cursor movement
        cursor_speed = 15
        if movement == 1: self.cursor_pos.y -= cursor_speed
        elif movement == 2: self.cursor_pos.y += cursor_speed
        elif movement == 3: self.cursor_pos.x -= cursor_speed
        elif movement == 4: self.cursor_pos.x += cursor_speed
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.GAME_HEIGHT)

        # Cycle ship selection
        if shift_pressed:
            self.selected_ship_type_idx = (self.selected_ship_type_idx + 1) % len(self.available_ship_types)
            # sfx: UI_bleep

        # Deploy ship / Activate combo
        if space_pressed:
            if self.armed_combo:
                self.score += 5 # Combo activation reward
                self._create_explosion(self.armed_combo['pos'], 100, 50, (255, 255, 100))
                for enemy in self.enemy_ships:
                    if enemy.pos.distance_to(self.armed_combo['pos']) < 150:
                        is_destroyed = enemy.take_damage(100)
                        self.score += 0.1 # damage reward
                        if is_destroyed: self.score += 1.0 # destroy reward
                self.armed_combo = None
                # sfx: combo_blast
            else:
                ship_name = self.available_ship_types[self.selected_ship_type_idx]
                ship_info = self.SHIP_TYPES[ship_name]
                new_ship = Ship(self.cursor_pos.copy(), ship_name, ship_info['color'], 'player')
                self.player_ships.append(new_ship)
                self._create_explosion(self.cursor_pos, 20, 5, (200, 200, 255)) # Warp-in effect
                # sfx: ship_warp_in

    def _fire_ship(self, ship):
        if not ship.target: return 0
        
        proj_vel = (ship.target.pos - ship.pos).normalize() * ship.proj_speed
        proj = Projectile(ship.pos.copy(), proj_vel, ship.color, ship.proj_damage)
        
        if ship.team == 'player':
            self.player_projectiles.append(proj)
        else:
            self.enemy_projectiles.append(proj)
        
        ship.cooldown = ship.fire_rate
        # sfx: laser_shoot
        return 0

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Enemies
        for p in self.player_projectiles:
            for e in self.enemy_ships:
                if p.pos.distance_to(e.pos) < 15: # Collision radius
                    is_destroyed = e.take_damage(p.damage)
                    reward += 0.1 # Damage reward
                    if is_destroyed:
                        reward += 1.0 # Destroy reward
                        self._create_explosion(e.pos, 50, 15)
                        # sfx: explosion
                    p.lifetime = 0 # Mark for cleanup
                    break
        
        # Enemy projectiles vs Players
        for p in self.enemy_projectiles:
            for player_ship in self.player_ships:
                if p.pos.distance_to(player_ship.pos) < 15:
                    is_destroyed = player_ship.take_damage(p.damage)
                    reward -= 0.1 # Penalty for being hit
                    if is_destroyed:
                        self._create_explosion(player_ship.pos, 50, 15)
                        # sfx: explosion
                    p.lifetime = 0
                    break
        return reward
    
    def _cleanup_entities(self):
        self.player_ships = [s for s in self.player_ships if s.health > 0]
        self.enemy_ships = [s for s in self.enemy_ships if s.health > 0]
        self.player_projectiles = [p for p in self.player_projectiles if p.lifetime > 0]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if p.lifetime > 0]
        self.particles = [p for p in self.particles if p.lifetime > 0]

    def _start_next_wave(self):
        self.wave += 1
        
        # Unlock new ships
        if self.wave == 5 and 'interceptor' not in self.available_ship_types:
            self.available_ship_types.append('interceptor')
        if self.wave == 10 and 'bomber' not in self.available_ship_types:
            self.available_ship_types.append('bomber')

        num_enemies = 2 + self.wave
        for _ in range(num_enemies):
            x = random.uniform(0, self.WIDTH)
            y = random.uniform(-50, -20)
            pos = pygame.Vector2(x, y)
            enemy = Ship(pos, 'fighter', self.COLOR_ENEMY, 'enemy')
            
            # Assert that enemy ship health increases by exactly 5% per wave.
            base_health = enemy.max_health
            enemy.max_health = base_health * (1.05 ** (self.wave - 1))
            enemy.health = enemy.max_health
            
            self.enemy_ships.append(enemy)

    def _check_combos(self):
        # Reset combo participant flags
        for ship in self.player_ships:
            ship.is_combo_participant = False
        self.armed_combo = None

        # Simple 2-fighter combo
        fighters = [s for s in self.player_ships if s.ship_type == 'fighter']
        if len(fighters) >= 2:
            for i in range(len(fighters)):
                for j in range(i + 1, len(fighters)):
                    ship1 = fighters[i]
                    ship2 = fighters[j]
                    if ship1.pos.distance_to(ship2.pos) < 80:
                        ship1.is_combo_participant = True
                        ship2.is_combo_participant = True
                        combo_pos = (ship1.pos + ship2.pos) / 2
                        self.armed_combo = {'type': 'fusion_beam', 'pos': combo_pos}
                        return # Arm only one combo at a time

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave}

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for star in self.stars:
            star['pos'].y += star['speed']
            if star['pos'].y > self.HEIGHT:
                star['pos'].y = 0
                star['pos'].x = random.uniform(0, self.WIDTH)
            pygame.draw.circle(self.screen, star['color'], (int(star['pos'].x), int(star['pos'].y)), star['size'])

    def _render_game(self):
        # Draw combo armed indicator
        if self.armed_combo:
            pos = self.armed_combo['pos']
            radius = 30 + 10 * math.sin(self.steps * 0.2)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), (255, 255, 0, 100))

        # Draw entities
        for p in self.particles: p.draw(self.screen)
        for proj in self.player_projectiles + self.enemy_projectiles: proj.draw(self.screen)
        for ship in self.player_ships + self.enemy_ships: ship.draw(self.screen)

        # Draw cursor
        cs = 8 # cursor size
        pos = self.cursor_pos
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos.x - cs, pos.y), (pos.x + cs, pos.y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos.x, pos.y - cs), (pos.x, pos.y + cs), 2)

    def _render_ui(self):
        # Panel
        ui_rect = pygame.Rect(0, self.GAME_HEIGHT, self.WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, ui_rect, 2)

        # Score and Wave
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        wave_text = self.font_large.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, self.GAME_HEIGHT + 15))
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 20, self.GAME_HEIGHT + 15))

        # Selected Ship
        ship_name = self.available_ship_types[self.selected_ship_type_idx]
        ship_info = self.SHIP_TYPES[ship_name]
        select_text = self.font_small.render(f"Deploy: {ship_name.upper()}", True, self.COLOR_TEXT)
        self.screen.blit(select_text, (self.WIDTH/2 - select_text.get_width()/2, self.GAME_HEIGHT + 10))
        
        # Combo status
        combo_status = "COMBO: ARMED" if self.armed_combo else "COMBO: ---"
        combo_color = (255, 255, 0) if self.armed_combo else self.COLOR_TEXT
        combo_text = self.font_small.render(combo_status, True, combo_color)
        self.screen.blit(combo_text, (self.WIDTH/2 - combo_text.get_width()/2, self.GAME_HEIGHT + 35))


    def _generate_stars(self):
        self.stars.clear()
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)),
                'size': random.choice([1, 1, 1, 2]),
                'speed': random.uniform(0.1, 0.5),
                'color': random.choice([(50, 50, 80), (80, 80, 100), (30, 30, 50)])
            })

    def _create_explosion(self, pos, num_particles, size, color=(255, 150, 50)):
        for _ in range(num_particles):
            angle = random.uniform(0, 360)
            speed = random.uniform(1, 5)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            p_size = random.uniform(size/2, size)
            lifetime = random.randint(20, 40)
            self.particles.append(Particle(pos, vel, p_size, color, lifetime))

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # To run with display, comment out the os.environ line at the top of the file
    try:
        is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    except (KeyError, NameError):
        is_headless = False

    if is_headless:
        print("Running in headless mode. No display will be shown.")
        env = GameEnv()
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished.")
                obs, info = env.reset()
        env.close()

    else:
        env = GameEnv(render_mode="rgb_array")
        pygame.display.set_caption("Combo Space Battle")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        
        obs, info = env.reset()
        done = False
        
        while not done:
            # Map keyboard keys to the MultiDiscrete action space
            keys = pygame.key.get_pressed()
            
            movement = 0 # none
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            env.clock.tick(30) # Run at 30 FPS for smooth gameplay feel

        env.close()