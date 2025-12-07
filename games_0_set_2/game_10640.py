import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:42:34.759982
# Source Brief: brief_00640.md
# Brief Index: 640
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
        "Survive waves of enemies in a neon arena. Collect parts from fallen foes to craft powerful new weapons."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to shoot and shift to switch weapons."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 3000  # Increased for longer games
        self.MAX_WAVES = 20

        # Colors
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_GRID = (30, 10, 50)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 128, 255)
        self.ENEMY_COLORS = {
            "melee": ((255, 50, 50), (128, 0, 0)),        # Red
            "ranged": ((50, 100, 255), (0, 50, 128)),     # Blue
            "healer": ((50, 255, 100), (0, 128, 50)),     # Green
            "special": ((200, 50, 255), (100, 0, 128)),   # Purple
        }
        self.PART_COLORS = {
            "red": self.ENEMY_COLORS["melee"][0],
            "blue": self.ENEMY_COLORS["ranged"][0],
            "green": self.ENEMY_COLORS["healer"][0],
            "purple": self.ENEMY_COLORS["special"][0],
        }
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_HEALTH_HIGH = (0, 255, 128)
        self.COLOR_HEALTH_LOW = (255, 50, 50)

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("consolas", 16)
            self.font_wave = pygame.font.SysFont("impact", 48)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 20)
            self.font_wave = pygame.font.SysFont(None, 60)

        # Weapon Definitions
        self.WEAPONS = {
            "pistol": {"fire_rate": 10, "damage": 10, "speed": 10, "ammo": float('inf'), "spread": 0.1, "projectiles": 1, "color": self.COLOR_WHITE},
            "shotgun": {"fire_rate": 25, "damage": 8, "speed": 8, "ammo": 20, "spread": 0.6, "projectiles": 6, "color": self.PART_COLORS["red"]},
            "rifle": {"fire_rate": 15, "damage": 25, "speed": 15, "ammo": 30, "spread": 0.0, "projectiles": 1, "color": self.PART_COLORS["blue"]},
            "beam": {"fire_rate": 5, "damage": 5, "speed": 20, "ammo": 50, "spread": 0.05, "projectiles": 1, "color": self.PART_COLORS["purple"]},
        }
        self.RECIPES = {
            "shotgun": {"parts": {"red": 5}, "unlock_wave": 1},
            "rifle": {"parts": {"blue": 8}, "unlock_wave": 5},
            "beam": {"parts": {"purple": 3, "blue": 5}, "unlock_wave": 10},
        }

        # Initialize state variables
        self.reset()
        
        # Self-check
        # self.validate_implementation() # Comment out for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Player State
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = 100
        self.player_max_health = 100
        self.player_speed = 4
        self.player_size = 12
        self.player_facing_direction = pygame.Vector2(0, -1)
        self.player_weapons = ["pistol"]
        self.player_weapon_idx = 0
        self.player_weapon_cooldown = 0
        self.player_inventory = { "red": 0, "blue": 0, "green": 0, "purple": 0 }
        self.player_ammo = {w: self.WEAPONS[w]["ammo"] for w in self.WEAPONS}

        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.wave_transition_timer = 120 # 2 seconds at 60fps, but we use it as a counter
        self.prev_shift_held = False
        
        # Entity Lists
        self.enemies = []
        self.projectiles = []
        self.parts_on_ground = []
        self.particles = []

        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # Handle wave transitions
        if not self.enemies and not self.game_over:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer <= 0:
                reward += self._start_new_wave()
        else:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # Update player state based on actions
            reward += self._handle_player_actions(movement, space_held, shift_held)
            
            # Update game world
            self._update_entities()
            reward += self._handle_collisions()

        # Check for termination conditions
        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
        elif self.wave_number > self.MAX_WAVES:
            reward += 100
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Gymnasium standard is to set both to True on truncation

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_player_actions(self, movement, space_held, shift_held):
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.player_speed
            self.player_facing_direction = move_vec.copy()
        
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)

        # Weapon Cooldown
        if self.player_weapon_cooldown > 0:
            self.player_weapon_cooldown -= 1

        # Shooting
        if space_held and self.player_weapon_cooldown == 0:
            self._fire_weapon()

        # Weapon Switching
        if shift_held and not self.prev_shift_held and len(self.player_weapons) > 1:
            self.player_weapon_idx = (self.player_weapon_idx + 1) % len(self.player_weapons)
            # Sfx: Weapon switch
        self.prev_shift_held = shift_held
        return 0

    def _fire_weapon(self):
        weapon_name = self.player_weapons[self.player_weapon_idx]
        weapon = self.WEAPONS[weapon_name]

        if self.player_ammo[weapon_name] <= 0:
            # Sfx: Empty clip
            return

        self.player_weapon_cooldown = weapon["fire_rate"]
        self.player_ammo[weapon_name] -= 1
        # Sfx: Weapon fire sound based on type

        target_dir = self._get_target_direction()

        for _ in range(weapon["projectiles"]):
            spread_angle = random.uniform(-weapon["spread"], weapon["spread"])
            fire_dir = target_dir.rotate(math.degrees(spread_angle))
            
            self.projectiles.append({
                "pos": self.player_pos.copy(),
                "vel": fire_dir * weapon["speed"],
                "damage": weapon["damage"],
                "color": weapon["color"],
                "lifespan": 40
            })

    def _get_target_direction(self):
        if not self.enemies:
            return self.player_facing_direction.copy()

        best_target = None
        min_dist_sq = float('inf')
        
        for enemy in self.enemies:
            vec_to_enemy = enemy["pos"] - self.player_pos
            dist_sq = vec_to_enemy.length_squared()
            
            if dist_sq == 0: continue
            
            # Check if enemy is in a 120-degree forward cone
            if self.player_facing_direction.dot(vec_to_enemy.normalize()) > -0.5: # cos(120/2) = cos(60) = 0.5, but let's use a wider angle
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_target = vec_to_enemy
        
        if best_target:
            return best_target.normalize()
        return self.player_facing_direction.copy()

    def _update_entities(self):
        # Update projectiles
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0 or not (0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT):
                self.projectiles.remove(p)

        # Update enemies
        for e in self.enemies:
            if e["type"] == "ranged":
                # Keep distance
                dist_to_player = self.player_pos.distance_to(e["pos"])
                direction = (self.player_pos - e["pos"]).normalize() if dist_to_player > 0 else pygame.Vector2(1,0)
                if dist_to_player < 200:
                    e["pos"] -= direction * e["speed"]
                elif dist_to_player > 250:
                    e["pos"] += direction * e["speed"]
                # TODO: Add ranged attack logic
            else: # melee
                direction = self.player_pos - e["pos"]
                if direction.length() > 0:
                    direction.normalize_ip()
                e["pos"] += direction * e["speed"]
            
            e["pos"].x = np.clip(e["pos"].x, 0, self.WIDTH)
            e["pos"].y = np.clip(e["pos"].y, 0, self.HEIGHT)

        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        # Projectile-Enemy collisions
        for p in self.projectiles[:]:
            for e in self.enemies[:]:
                if p["pos"].distance_to(e["pos"]) < e["size"]:
                    e["health"] -= p["damage"]
                    reward += 0.1 # Hit reward
                    self._create_particles(p["pos"], p["color"], 5)
                    if p in self.projectiles: self.projectiles.remove(p)
                    # Sfx: Hit sound
                    
                    if e["health"] <= 0:
                        reward += 1.0 # Kill reward
                        self._create_particles(e["pos"], e["color"][0], 20, 3)
                        self._drop_parts(e)
                        self.enemies.remove(e)
                        # Sfx: Enemy explosion
                    break
        
        # Player-Enemy collisions
        for e in self.enemies:
            if self.player_pos.distance_to(e["pos"]) < self.player_size + e["size"] / 2:
                self.player_health -= 10
                reward -= 0.1 # Damage penalty
                self._create_particles(self.player_pos, self.COLOR_HEALTH_LOW, 10)
                self.player_health = max(0, self.player_health)
                # Sfx: Player hurt

        # Player-Part collisions
        for part in self.parts_on_ground[:]:
            if self.player_pos.distance_to(part["pos"]) < self.player_size + 5:
                self.player_inventory[part["type"]] += 1
                if part["type"] == "green":
                    self.player_health = min(self.player_max_health, self.player_health + 10)
                self.parts_on_ground.remove(part)
                # Sfx: Part pickup
        
        return reward

    def _drop_parts(self, enemy):
        part_type = "red"
        if enemy["type"] == "ranged": part_type = "blue"
        elif enemy["type"] == "healer": part_type = "green"
        elif enemy["type"] == "special": part_type = "purple"
        
        num_parts = 1 if self.np_random.random() < 0.7 else 2
        for _ in range(num_parts):
            offset = pygame.Vector2(self.np_random.uniform(-10, 10), self.np_random.uniform(-10, 10))
            self.parts_on_ground.append({
                "pos": enemy["pos"] + offset,
                "type": part_type,
                "color": self.PART_COLORS[part_type]
            })

    def _start_new_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return 0
            
        self.wave_transition_timer = 90
        self._spawn_enemies()
        craft_reward = self._attempt_crafting()
        return 10.0 + craft_reward # Wave completion reward

    def _spawn_enemies(self):
        num_enemies = 3 + self.wave_number
        for _ in range(num_enemies):
            # Spawn off-screen
            edge = self.np_random.integers(4)
            if edge == 0: x, y = self.np_random.uniform(-20, self.WIDTH+20), -20
            elif edge == 1: x, y = self.np_random.uniform(-20, self.WIDTH+20), self.HEIGHT+20
            elif edge == 2: x, y = -20, self.np_random.uniform(-20, self.HEIGHT+20)
            else: x, y = self.WIDTH+20, self.np_random.uniform(-20, self.HEIGHT+20)
            
            enemy_type_roll = self.np_random.random()
            if enemy_type_roll < 0.6: e_type = "melee"
            elif enemy_type_roll < 0.9: e_type = "ranged"
            else: e_type = "healer"
            # TODO: Add special enemies in later waves
            
            health_mult = 1 + (0.05 * (self.wave_number-1))
            speed_mult = 1 + (0.02 * (self.wave_number-1))

            self.enemies.append({
                "pos": pygame.Vector2(x, y),
                "type": e_type,
                "health": 20 * health_mult,
                "max_health": 20 * health_mult,
                "speed": self.np_random.uniform(1.0, 1.5) * speed_mult,
                "size": 15,
                "color": self.ENEMY_COLORS[e_type]
            })

    def _attempt_crafting(self):
        reward = 0
        for weapon_name, recipe in self.RECIPES.items():
            if weapon_name not in self.player_weapons and self.wave_number >= recipe["unlock_wave"]:
                can_craft = True
                for part_type, required_amount in recipe["parts"].items():
                    if self.player_inventory[part_type] < required_amount:
                        can_craft = False
                        break
                if can_craft:
                    for part_type, required_amount in recipe["parts"].items():
                        self.player_inventory[part_type] -= required_amount
                    self.player_weapons.append(weapon_name)
                    reward += 5.0 # Crafting reward
                    # Sfx: Craft success
        return reward

    def _create_particles(self, pos, color, count, speed_max=2):
        for _ in range(count):
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            if vel.length() > 0: vel.normalize_ip()
            vel *= self.np_random.uniform(0.5, speed_max)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(10, 20),
                "color": color,
                "size": self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.player_health,
            "inventory": self.player_inventory,
            "enemies": len(self.enemies),
        }

    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        for p in self.particles: self._render_particle(p)
        for p in self.parts_on_ground: self._render_part(p)
        for p in self.projectiles: self._render_projectile(p)
        for e in self.enemies: self._render_enemy(e)
        self._render_player()
        self._render_ui()

    def _render_background(self):
        for i in range(0, self.WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)
    
    def _render_player(self):
        p = self.player_pos
        s = self.player_size
        angle = self.player_facing_direction.angle_to(pygame.Vector2(0, -1))
        
        points = [
            pygame.Vector2(0, -s).rotate(-angle) + p,
            pygame.Vector2(-s*0.7, s*0.7).rotate(-angle) + p,
            pygame.Vector2(s*0.7, s*0.7).rotate(-angle) + p,
        ]
        int_points = [(int(pt.x), int(pt.y)) for pt in points]
        self._draw_glowing_polygon(self.screen, int_points, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _render_enemy(self, enemy):
        color, glow_color = enemy["color"]
        pos = (int(enemy["pos"].x), int(enemy["pos"].y))
        size = int(enemy["size"])
        if enemy["type"] == "melee":
            points = [ (pos[0]-size, pos[1]-size), (pos[0]+size, pos[1]-size), (pos[0]+size, pos[1]+size), (pos[0]-size, pos[1]+size) ]
            self._draw_glowing_polygon(self.screen, points, color, glow_color)
        else: # Ranged, Healer, etc.
            self._draw_glowing_circle(self.screen, pos, size, color, glow_color)

    def _render_projectile(self, p):
        pos = (int(p["pos"].x), int(p["pos"].y))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, p["color"])
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, p["color"])
        # Trail effect
        trail_pos = p["pos"] - p["vel"] * 0.5
        pygame.gfxdraw.filled_circle(self.screen, int(trail_pos.x), int(trail_pos.y), 2, (*p["color"], 128))

    def _render_part(self, part):
        pos = (int(part["pos"].x), int(part["pos"].y))
        color = part["color"]
        points = [(pos[0], pos[1]-4), (pos[0]+4, pos[1]), (pos[0], pos[1]+4), (pos[0]-4, pos[1])]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_particle(self, p):
        alpha = int(255 * (p["lifespan"] / 20.0))
        color = (*p["color"], alpha)
        size = int(p["size"] * (p["lifespan"] / 20.0))
        if size > 0:
            pygame.draw.rect(self.screen, color, (int(p["pos"].x), int(p["pos"].y), size, size))

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.player_max_health
        health_color = (
            int(self.COLOR_HEALTH_LOW[0] * (1-health_ratio) + self.COLOR_HEALTH_HIGH[0] * health_ratio),
            int(self.COLOR_HEALTH_LOW[1] * (1-health_ratio) + self.COLOR_HEALTH_HIGH[1] * health_ratio),
            int(self.COLOR_HEALTH_LOW[2] * (1-health_ratio) + self.COLOR_HEALTH_HIGH[2] * health_ratio),
        )
        bar_width = 200
        pygame.draw.rect(self.screen, (50,50,50), (self.WIDTH/2 - bar_width/2 - 2, self.HEIGHT - 30 - 2, bar_width+4, 20+4))
        pygame.draw.rect(self.screen, self.COLOR_BG, (self.WIDTH/2 - bar_width/2, self.HEIGHT - 30, bar_width, 20))
        pygame.draw.rect(self.screen, health_color, (self.WIDTH/2 - bar_width/2, self.HEIGHT - 30, bar_width * health_ratio, 20))

        # Score and Wave
        self._draw_text(f"SCORE: {int(self.score)}", 16, self.COLOR_WHITE, (10, 10), "topleft")
        if not self.enemies and self.wave_transition_timer > 0 and not self.game_over:
             self._draw_text(f"WAVE {self.wave_number} CLEARED", 48, self.COLOR_PLAYER, (self.WIDTH/2, self.HEIGHT/2 - 50))
             self._draw_text(f"NEXT WAVE IN {self.wave_transition_timer//30 + 1}", 24, self.COLOR_WHITE, (self.WIDTH/2, self.HEIGHT/2))
        else:
             self._draw_text(f"WAVE: {self.wave_number}", 16, self.COLOR_WHITE, (self.WIDTH-10, 10), "topright")

        # Weapon Info
        weapon_name = self.player_weapons[self.player_weapon_idx]
        ammo = self.player_ammo[weapon_name]
        ammo_str = f"{int(ammo)}" if ammo != float('inf') else "∞"
        self._draw_text(f"{weapon_name.upper()}", 16, self.COLOR_WHITE, (self.WIDTH-10, self.HEIGHT-35), "bottomright")
        self._draw_text(f"AMMO: {ammo_str}", 16, self.COLOR_WHITE, (self.WIDTH-10, self.HEIGHT-15), "bottomright")
        
        # Parts Inventory
        y_offset = 0
        for part_type, count in self.player_inventory.items():
            if count > 0:
                self._draw_text(f"{count}", 16, self.PART_COLORS[part_type], (25, self.HEIGHT - 20 - y_offset), "midleft")
                pygame.draw.rect(self.screen, self.PART_COLORS[part_type], (10, self.HEIGHT - 25 - y_offset, 10, 10))
                y_offset += 20

    def _draw_glowing_polygon(self, surf, points, color, glow_color):
        for i in range(4, 0, -1):
            inflated_points = []
            # This is a simplified inflation, not geometrically perfect but fast
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            for p in points:
                dx, dy = p[0] - cx, p[1] - cy
                length = math.hypot(dx, dy)
                if length > 0:
                    inflated_points.append((p[0] + dx/length * i, p[1] + dy/length * i))
                else:
                    inflated_points.append(p)
            
            glow_alpha_color = (*glow_color, 40)
            pygame.gfxdraw.aapolygon(surf, inflated_points, glow_alpha_color)
            pygame.gfxdraw.filled_polygon(surf, inflated_points, glow_alpha_color)
        
        pygame.gfxdraw.aapolygon(surf, points, color)
        pygame.gfxdraw.filled_polygon(surf, points, color)

    def _draw_glowing_circle(self, surf, center, radius, color, glow_color):
        for i in range(5, 0, -1):
            glow_alpha_color = (*glow_color, 30)
            pygame.gfxdraw.aacircle(surf, center[0], center[1], radius + i, glow_alpha_color)
        pygame.gfxdraw.filled_circle(surf, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surf, center[0], center[1], radius, color)

    def _draw_text(self, text, size, color, pos, align="center"):
        # Re-create font object if size changes to avoid aliasing issues
        if not hasattr(self, f"font_{size}"):
            try:
                setattr(self, f"font_{size}", pygame.font.SysFont("consolas", size))
            except pygame.error:
                setattr(self, f"font_{size}", pygame.font.SysFont(None, int(size*1.2)))
        
        font = getattr(self, f"font_{size}")
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        setattr(text_rect, align, pos)
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Example usage
    env = GameEnv()
    obs, info = env.reset()
    
    # Let's run a few random steps to demonstrate it works
    print("Running 100 random steps...")
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print("Episode finished early.")
            break
    print(f"Finished. Total reward: {total_reward}")
    print(f"Final info: {info}")
    
    # To visualize, you would need to blit the numpy array to a pygame display
    # This requires running it in a context where a display is created.
    # Example visualization loop:
    print("\nStarting visualization loop (press ESC to quit)...")
    env.reset()
    # Unset the dummy video driver to allow for a display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    pygame.display.init()
    pygame.display.set_caption("Neon Arena")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    done = False
    while not done:
        # Simple manual control mapping for testing
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}. Press 'R' to restart.")

        clock.tick(60) # Run at 60 FPS for smooth manual play
        
    env.close()