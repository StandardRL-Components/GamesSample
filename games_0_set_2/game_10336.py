import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:12:54.549316
# Source Brief: brief_00336.md
# Brief Index: 336
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# Helper function for drawing glowing shapes
def draw_glowing_circle(surface, color, center, radius, max_glow=15):
    """Draws a circle with a glowing aura."""
    base_color = pygame.Color(color)
    for i in range(max_glow, 0, -2):
        glow_color = base_color.lerp((0, 0, 0, 0), i / max_glow)
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius + i), glow_color)
    pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), base_color)
    pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), base_color)

# Helper function for drawing glowing polygons
def draw_glowing_polygon(surface, color, points, max_glow=10):
    """Draws a polygon with a glowing aura."""
    base_color = pygame.Color(color)
    
    # Draw glow layers
    for i in range(max_glow, 0, -2):
        glow_color = base_color.lerp((0, 0, 0, 0), i / max_glow)
        
        # Expand the polygon for the glow effect
        glow_points = []
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        
        for p in points:
            dx, dy = p[0] - cx, p[1] - cy
            dist = math.hypot(dx, dy)
            if dist > 0:
                glow_points.append((p[0] + dx/dist * i, p[1] + dy/dist * i))
            else:
                glow_points.append(p)
                
        pygame.gfxdraw.filled_polygon(surface, glow_points, glow_color)

    # Draw the main polygon
    pygame.gfxdraw.aapolygon(surface, points, base_color)
    pygame.gfxdraw.filled_polygon(surface, points, base_color)

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, size, velocity):
        self.pos = pygame.Vector2(x, y)
        self.color = color
        self.life = life
        self.initial_life = life
        self.size = size
        self.velocity = pygame.Vector2(velocity)

    def update(self):
        self.pos += self.velocity
        self.life -= 1
        self.velocity *= 0.95 # Damping

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            color = self.color[:3] + (alpha,)
            radius = int(self.size * (self.life / self.initial_life))
            if radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), radius, color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Command a squadron of clones in a top-down shooter. Deploy new units, "
        "magnetize enemy weapons, and guide your leader to the objective."
    )
    user_guide = (
        "Controls: Use ↑↓←→ to move. Press Space to deploy a clone and Shift to "
        "magnetize and steal enemy weapons."
    )
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_LEADER = (0, 255, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_WEAPON = (100, 150, 255)
        self.COLOR_OBJECTIVE = (255, 220, 0)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_PROJECTILE_PLAYER = (0, 255, 200)
        self.COLOR_PROJECTILE_ENEMY = (255, 100, 100)

        # Game constants
        self.MAX_STEPS = 2000
        self.PLAYER_SPEED = 20
        self.ENEMY_SPEED = 10
        self.MAX_CLONES = 10
        self.CLONE_DEPLOY_COST = 5
        self.MAGNETIZE_RADIUS = 100
        
        # Initialize state variables
        self.leader = None
        self.clones = None
        self.enemies = None
        self.particles = None
        self.magnetize_effects = None
        self.weapon_drops = None
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is done by the wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.leader = {
            "pos": pygame.Vector2(50, self.HEIGHT // 2),
            "size": 10,
            "health": 100,
            "max_health": 100,
            "weapon": {"name": "Pistol", "range": 150, "damage": 10, "cooldown": 0, "max_cooldown": 10},
            "is_leader": True
        }
        self.clones = [self.leader]
        self.enemies = []
        self.particles = deque(maxlen=500)
        self.magnetize_effects = []
        self.weapon_drops = []
        
        self.clone_energy = 5
        self.max_clone_energy = 10

        self.objective_pos = pygame.Vector2(self.WIDTH - 50, self.np_random.integers(50, self.HEIGHT - 50))
        
        self._spawn_enemies(3)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.game_over = False

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- PLAYER ACTIONS ---
        if shift_pressed:
            reward += self._handle_magnetize()
        
        if space_pressed:
            reward += self._handle_deploy()

        self._handle_movement(movement)

        # --- AI & WORLD TURN ---
        reward += self._update_clones_and_enemies()
        self._update_particles_and_effects()
        self._spawn_new_enemies()
        
        # Replenish clone energy
        if self.steps % 5 == 0 and self.clone_energy < self.max_clone_energy:
            self.clone_energy += 1

        self.steps += 1
        
        # --- TERMINATION & REWARD ---
        terminated = False
        if self._check_objective_reached():
            reward += 100
            terminated = True
            # SFX: Win Jingle
        elif not self.clones or self.leader is None:
            reward -= 100
            terminated = True
            # SFX: Loss Buzzer
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement_action):
        if not self.leader:
            return
            
        move_vec = pygame.Vector2(0, 0)
        if movement_action == 1: move_vec.y = -1 # Up
        elif movement_action == 2: move_vec.y = 1 # Down
        elif movement_action == 3: move_vec.x = -1 # Left
        elif movement_action == 4: move_vec.x = 1 # Right
        
        if move_vec.length() > 0:
            self.leader['pos'] += move_vec * self.PLAYER_SPEED
            # SFX: Player move woosh
        
        # Clamp leader position to screen bounds
        self.leader['pos'].x = np.clip(self.leader['pos'].x, self.leader['size'], self.WIDTH - self.leader['size'])
        self.leader['pos'].y = np.clip(self.leader['pos'].y, self.leader['size'], self.HEIGHT - self.leader['size'])

    def _handle_magnetize(self):
        if not self.leader:
            return 0
            
        reward = 0
        magnetized_something = False
        enemies_in_range = [e for e in self.enemies if self.leader['pos'].distance_to(e['pos']) < self.MAGNETIZE_RADIUS and e.get('weapon')]
        
        stolen_weapons = []
        for enemy in enemies_in_range:
            stolen_weapons.append(enemy['weapon'])
            enemy['weapon'] = None # Disarm enemy
            # SFX: Weapon disarm zap
            self._create_particles(enemy['pos'], self.COLOR_WEAPON, 10, 2)

        if stolen_weapons:
            magnetized_something = True
            self.leader['weapon'] = random.choice(stolen_weapons)
            self.leader['weapon']['cooldown'] = 0 # Ready to fire
            reward += 1.0 * len(stolen_weapons)
            # SFX: Magnetize success chime

        if magnetized_something:
            self.magnetize_effects.append({"pos": self.leader['pos'], "radius": 0, "life": 20})
        return reward

    def _handle_deploy(self):
        if not self.leader:
            return 0
            
        if self.clone_energy >= self.CLONE_DEPLOY_COST and len(self.clones) < self.MAX_CLONES:
            self.clone_energy -= self.CLONE_DEPLOY_COST
            
            offset = pygame.Vector2(self.np_random.uniform(-40, 40), self.np_random.uniform(-40, 40))
            new_pos = self.leader['pos'] + offset
            new_pos.x = np.clip(new_pos.x, 8, self.WIDTH - 8)
            new_pos.y = np.clip(new_pos.y, 8, self.HEIGHT - 8)

            new_clone = {
                "pos": new_pos, "size": 8, "health": 50, "max_health": 50, "is_leader": False,
                "weapon": {"name": "Turret", "range": 200, "damage": 8, "cooldown": 0, "max_cooldown": 15},
            }
            self.clones.append(new_clone)
            self._create_particles(new_clone['pos'], self.COLOR_PLAYER, 20, 3)
            # SFX: Clone deploy spawn
            return 0 # No immediate reward for deploying
        # SFX: Deploy fail buzz
        return 0

    def _update_clones_and_enemies(self):
        reward = 0
        # --- Attacks ---
        projectiles = []
        # Clones attack enemies
        for clone in self.clones:
            if clone['weapon']['cooldown'] > 0:
                clone['weapon']['cooldown'] -= 1
                continue
            
            # Find closest enemy in range
            target = None
            min_dist = clone['weapon']['range']
            for enemy in self.enemies:
                dist = clone['pos'].distance_to(enemy['pos'])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                projectiles.append({"start": clone['pos'], "end": target['pos'], "color": self.COLOR_PROJECTILE_PLAYER, "damage": clone['weapon']['damage']})
                clone['weapon']['cooldown'] = clone['weapon']['max_cooldown']
                # SFX: Player shoot laser
        
        # Enemies attack clones
        for enemy in self.enemies:
            # Find closest clone
            target = None
            min_dist = float('inf')
            for clone in self.clones:
                dist = enemy['pos'].distance_to(clone['pos'])
                if dist < min_dist:
                    min_dist = dist
                    target = clone

            if not target: continue

            # Move towards target
            if (target['pos'] - enemy['pos']).length() > 0:
                direction = (target['pos'] - enemy['pos']).normalize()
                enemy['pos'] += direction * self.ENEMY_SPEED
            
            # Attack if in range
            if enemy.get('weapon') and min_dist < enemy['weapon']['range']:
                if enemy['weapon']['cooldown'] > 0:
                    enemy['weapon']['cooldown'] -= 1
                else:
                    projectiles.append({"start": enemy['pos'], "end": target['pos'], "color": self.COLOR_PROJECTILE_ENEMY, "damage": enemy['weapon']['damage']})
                    enemy['weapon']['cooldown'] = enemy['weapon']['max_cooldown']
                    # SFX: Enemy shoot laser
            elif not enemy.get('weapon') and min_dist < 20: # Melee attack if unarmed
                target['health'] -= 5
                self._create_particles(target['pos'], self.COLOR_ENEMY, 5, 1)

        # --- Resolve Projectile Damage ---
        for proj in projectiles:
            self._create_particles(proj['end'], proj['color'], 15, 2)
            # Find target entity (closest to projectile end)
            entities = self.clones + self.enemies
            if not entities: continue
            
            target_entity = min(entities, key=lambda e: e['pos'].distance_to(proj['end']))
            if target_entity['pos'].distance_to(proj['end']) < target_entity['size'] * 2:
                 target_entity['health'] -= proj['damage']

        # --- Cleanup Dead Entities ---
        clones_before = len(self.clones)
        self.clones = [c for c in self.clones if c['health'] > 0]
        if not any(c.get('is_leader', False) for c in self.clones):
            self.leader = None # Ensure leader is None if dead
        
        clones_lost = clones_before - len(self.clones)
        if clones_lost > 0:
            reward -= 0.1 * clones_lost
            # SFX: Clone destroyed explosion

        enemies_before = len(self.enemies)
        surviving_enemies = []
        for e in self.enemies:
            if e['health'] > 0:
                surviving_enemies.append(e)
            else:
                self._create_particles(e['pos'], self.COLOR_ENEMY, 30, 4)
                # SFX: Enemy destroyed explosion
        self.enemies = surviving_enemies
        enemies_killed = enemies_before - len(self.enemies)
        if enemies_killed > 0:
            reward += 0.1 * enemies_killed
            self.score += 10 * enemies_killed # Give points for kills

        return reward

    def _spawn_enemies(self, count):
        for _ in range(count):
            side = self.np_random.integers(4)
            if side == 0: x, y = self.WIDTH-20, self.np_random.uniform(20, self.HEIGHT-20)
            elif side == 1: x, y = 20, self.np_random.uniform(20, self.HEIGHT-20)
            elif side == 2: x, y = self.np_random.uniform(20, self.WIDTH-20), self.HEIGHT-20
            else: x, y = self.np_random.uniform(20, self.WIDTH-20), 20

            health_bonus = self.steps // 200
            enemy = {
                "pos": pygame.Vector2(x, y), "size": 8,
                "health": 20 + health_bonus, "max_health": 20 + health_bonus,
                "weapon": random.choice([
                    {"name": "E-Pistol", "range": 180, "damage": 5, "cooldown": 0, "max_cooldown": 20},
                    {"name": "E-Rifle", "range": 300, "damage": 8, "cooldown": 0, "max_cooldown": 30},
                ])
            }
            self.enemies.append(enemy)

    def _spawn_new_enemies(self):
        if self.steps > 0 and self.steps % 100 == 0:
            num_to_spawn = 1 + self.steps // 200
            self._spawn_enemies(num_to_spawn)

    def _check_objective_reached(self):
        if self.leader and self.leader['pos'].distance_to(self.objective_pos) < self.leader['size'] + 15:
            return True
        return False

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            velocity = (self.np_random.uniform(-max_speed, max_speed), self.np_random.uniform(-max_speed, max_speed))
            size = self.np_random.uniform(1, 4)
            life = self.np_random.integers(10, 30)
            self.particles.append(Particle(pos.x, pos.y, color, life, size, velocity))
            
    def _update_particles_and_effects(self):
        for p in list(self.particles):
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        for effect in list(self.magnetize_effects):
            effect['radius'] += self.MAGNETIZE_RADIUS / 20
            effect['life'] -= 1
            if effect['life'] <= 0:
                self.magnetize_effects.remove(effect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw objective
        draw_glowing_circle(self.screen, self.COLOR_OBJECTIVE, self.objective_pos, 15, 20)
        
        # Draw magnetize effects
        for effect in self.magnetize_effects:
            alpha = int(255 * (effect['life'] / 20))
            color = self.COLOR_WEAPON[:3] + (alpha,)
            pygame.gfxdraw.aacircle(self.screen, int(effect['pos'].x), int(effect['pos'].y), int(effect['radius']), color)

        # Draw enemies
        for enemy in self.enemies:
            points = [
                (enemy['pos'].x, enemy['pos'].y - enemy['size']),
                (enemy['pos'].x - enemy['size'], enemy['pos'].y + enemy['size']),
                (enemy['pos'].x + enemy['size'], enemy['pos'].y + enemy['size']),
            ]
            draw_glowing_polygon(self.screen, self.COLOR_ENEMY, points, 8)
            # Health bar
            if enemy['health'] < enemy['max_health']:
                self._draw_health_bar(enemy['pos'], enemy['health'], enemy['max_health'], enemy['size'])
        
        # Draw clones
        for clone in self.clones:
            color = self.COLOR_LEADER if clone.get('is_leader', False) else self.COLOR_PLAYER
            size = clone['size']
            if clone.get('is_leader', False):
                draw_glowing_circle(self.screen, color, clone['pos'], size, 15)
            else:
                 draw_glowing_circle(self.screen, color, clone['pos'], size, 10)
            # Health bar
            if clone['health'] < clone['max_health']:
                self._draw_health_bar(clone['pos'], clone['health'], clone['max_health'], size)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _draw_health_bar(self, pos, current_hp, max_hp, size):
        bar_width = size * 2
        bar_height = 4
        y_offset = size + 5
        
        bg_rect = pygame.Rect(pos.x - bar_width/2, pos.y - y_offset, bar_width, bar_height)
        
        health_ratio = max(0, current_hp / max_hp)
        fill_width = bar_width * health_ratio
        fill_rect = pygame.Rect(pos.x - bar_width/2, pos.y - y_offset, fill_width, bar_height)
        
        pygame.draw.rect(self.screen, (50, 50, 50), bg_rect)
        pygame.draw.rect(self.screen, (0, 200, 0) if health_ratio > 0.5 else (200, 200, 0) if health_ratio > 0.2 else (200, 0, 0), fill_rect)

    def _render_ui(self):
        # Clone Count
        clone_text = self.font_large.render(f"CLONES: {len(self.clones)}/{self.MAX_CLONES}", True, self.COLOR_TEXT)
        self.screen.blit(clone_text, (10, 10))

        # Clone Energy
        energy_text = self.font_small.render(f"ENERGY: {self.clone_energy}/{self.max_clone_energy}", True, self.COLOR_TEXT)
        self.screen.blit(energy_text, (10, 40))

        # Objective
        obj_text = self.font_large.render("OBJECTIVE: REACH YELLOW MARKER", True, self.COLOR_TEXT)
        self.screen.blit(obj_text, (self.WIDTH - obj_text.get_width() - 10, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 40))
        
        # Weapon
        if self.leader:
            weapon_text = self.font_small.render(f"WEAPON: {self.leader['weapon']['name']}", True, self.COLOR_TEXT)
            self.screen.blit(weapon_text, (10, self.HEIGHT - weapon_text.get_height() - 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "clones": len(self.clones),
            "enemies": len(self.enemies),
            "clone_energy": self.clone_energy,
            "leader_hp": self.leader['health'] if self.leader else 0,
        }

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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # This check is not needed in the final version but useful for dev
    try:
        env.validate_implementation()
    except Exception as e:
        print(f"Implementation validation failed: {e}")


    # Re-enable display for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Clone Command")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_pressed = 0
        shift_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1

        action = [movement, space_pressed, shift_pressed]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(10) # Run at 10 FPS for manual play

    env.close()