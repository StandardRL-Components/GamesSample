import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:50:52.335288
# Source Brief: brief_02789.md
# Brief Index: 2789
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent commands a squad of 4 mechs
    against waves of enemies, using timed and coordinated attacks to
    create powerful combos.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Command a squad of four mechs against waves of enemies. "
        "Use timed and coordinated attacks to create powerful combos and defeat your foes."
    )
    user_guide = (
        "Controls: 1-4 keys to select a mech. Press space to fire at the targeted enemy. "
        "Press shift to cycle between targets."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 30 * 60  # 60 seconds at 30 FPS

    # Colors
    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (30, 35, 50)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_UI_BAR_BG = (50, 50, 70)
    
    PROJECTILE_COLORS = {
        "NOUN": (255, 200, 0),
        "VERB": (0, 150, 255),
        "ADJECTIVE": (200, 100, 255),
        "ADVERB": (255, 100, 0),
    }
    MECH_TYPES = ["NOUN", "VERB", "ADJECTIVE", "ADVERB"]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_combo = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        self.mechs = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 0
        self.total_waves = 3
        
        self.selected_mech_idx = 0
        self.target_enemy_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.combo_timer = 0
        self.combo_multiplier = 1.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        # --- Initialize Game Entities ---
        self.mechs = []
        mech_y_positions = [80, 160, 240, 320]
        for i in range(4):
            mech_type = self.MECH_TYPES[i]
            pos = pygame.math.Vector2(50, mech_y_positions[i])
            self.mechs.append(Mech(pos, mech_type, self.PROJECTILE_COLORS[mech_type]))

        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.current_wave = 0
        self._spawn_wave()

        self.selected_mech_idx = 0
        self.target_enemy_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.combo_timer = 0
        self.combo_multiplier = 1.0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.game_over = self._check_termination()
        
        if not self.game_over:
            # --- Handle Actions ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # Mech selection
            if movement in [1, 2, 3, 4]:
                self.selected_mech_idx = movement - 1

            # Fire projectile (on rising edge of space key)
            if space_held and not self.prev_space_held:
                self._fire_projectile()

            # Cycle target (on rising edge of shift key)
            if shift_held and not self.prev_shift_held:
                self._cycle_target()

            self.prev_space_held = space_held
            self.prev_shift_held = shift_held

            # --- Update Game Logic ---
            self._update_entities()
            self._handle_collisions()
            self._cleanup_entities()
            self._update_combo()
            self._check_wave_completion()

        # Check for game over again after updates
        terminated = self._check_termination()
        if terminated and not self.game_over:
             # This means a terminal state was reached this step
            if all(m.health <= 0 for m in self.mechs):
                self.reward_this_step += -100 # Loss penalty
            elif self.current_wave > self.total_waves:
                self.reward_this_step += 100 # Win bonus
            self.game_over = True

        self.steps += 1
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info(),
        )

    def _spawn_wave(self):
        self.current_wave += 1
        if self.current_wave > self.total_waves:
            return

        num_enemies = 5
        base_health = 75 + self.current_wave * 25
        
        weakness_map = {1: "NOUN", 2: "VERB", 3: "ADJECTIVE"}
        weakness = weakness_map.get(self.current_wave, "ADVERB")

        for _ in range(num_enemies):
            pos = pygame.math.Vector2(
                self.np_random.uniform(self.SCREEN_WIDTH * 0.6, self.SCREEN_WIDTH * 0.9),
                self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            )
            self.enemies.append(Enemy(pos, base_health, weakness, self.np_random))

    def _fire_projectile(self):
        if not self.enemies:
            return
            
        mech = self.mechs[self.selected_mech_idx]
        if mech.cooldown_timer == 0:
            target = self.enemies[self.target_enemy_idx]
            # sfx: player_shoot.wav
            proj = mech.fire(target.pos)
            self.projectiles.append(proj)
            
            # Visual feedback for firing
            for _ in range(5):
                self.particles.append(Particle(
                    pos=mech.pos.copy(), 
                    color=mech.color, 
                    vel=pygame.math.Vector2(self.np_random.uniform(-2, 0), self.np_random.uniform(-1, 1)),
                    radius=self.np_random.uniform(2, 4),
                    lifespan=10
                ))

    def _cycle_target(self):
        if self.enemies:
            self.target_enemy_idx = (self.target_enemy_idx + 1) % len(self.enemies)

    def _update_entities(self):
        for mech in self.mechs:
            mech.update()
        for enemy in self.enemies:
            enemy.update()
        for proj in self.projectiles:
            proj.update()
        for particle in self.particles:
            particle.update()

    def _handle_collisions(self):
        for proj in self.projectiles:
            if not proj.active:
                continue
            for enemy in self.enemies:
                if not enemy.is_alive():
                    continue
                if proj.pos.distance_to(enemy.pos) < proj.radius + enemy.radius:
                    # sfx: enemy_hit.wav
                    proj.active = False
                    is_weak = proj.type == enemy.weakness
                    damage_multiplier = (2.0 if is_weak else 1.0)
                    
                    is_combo = self.combo_timer > 0
                    if is_combo:
                        # Combo hit
                        damage_multiplier *= 2.0 * self.combo_multiplier
                        self.combo_multiplier += 0.25
                        self.reward_this_step += 5
                    else:
                        # First hit of a potential combo
                        self.combo_multiplier = 1.0
                    
                    self.reward_this_step += 1
                    
                    damage = proj.damage * damage_multiplier
                    enemy.take_damage(damage)
                    
                    # Reset combo window timer
                    self.combo_timer = 15 # 0.5 seconds at 30fps
                    
                    # Create impact particles
                    for _ in range(15):
                        self.particles.append(Particle.create_explosion(proj.pos, proj.color, self.np_random))
                    break # Projectile can only hit one enemy

    def _cleanup_entities(self):
        self.projectiles = [p for p in self.projectiles if p.active and 0 < p.pos.x < self.SCREEN_WIDTH and 0 < p.pos.y < self.SCREEN_HEIGHT]
        
        dead_enemies = [e for e in self.enemies if not e.is_alive()]
        if dead_enemies:
            # sfx: enemy_explode.wav
            for enemy in dead_enemies:
                self.reward_this_step += 10
                self.score += 100
                for _ in range(40):
                    self.particles.append(Particle.create_explosion(enemy.pos, self.COLOR_ENEMY, self.np_random))
            self.enemies = [e for e in self.enemies if e.is_alive()]
            if not self.enemies:
                self.target_enemy_idx = 0
            elif self.target_enemy_idx >= len(self.enemies):
                self.target_enemy_idx = 0 # Target was destroyed, reset to first
        
        self.particles = [p for p in self.particles if p.is_alive()]

    def _update_combo(self):
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            if self.combo_multiplier > 1.0:
                self.score += int(10 * self.combo_multiplier)
            self.combo_multiplier = 1.0

    def _check_wave_completion(self):
        if not self.enemies and self.current_wave <= self.total_waves:
            self.reward_this_step += 20
            self.score += 200
            self._spawn_wave()

    def _check_termination(self):
        if self.current_wave > self.total_waves and not self.enemies:
            return True # Win condition
        if all(m.health <= 0 for m in self.mechs):
            return True # Lose condition
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render target line
        if self.enemies:
            mech = self.mechs[self.selected_mech_idx]
            target = self.enemies[self.target_enemy_idx]
            pygame.draw.line(self.screen, (100, 100, 100, 100), mech.pos, target.pos, 1)

        # Render entities
        for particle in self.particles:
            particle.draw(self.screen)
        for proj in self.projectiles:
            proj.draw(self.screen)
        for enemy in self.enemies:
            enemy.draw(self.screen)
        for i, mech in enumerate(self.mechs):
            is_selected = i == self.selected_mech_idx
            mech.draw(self.screen, is_selected)
            
        # Render target reticle
        if self.enemies:
            target = self.enemies[self.target_enemy_idx]
            reticle_color = self.PROJECTILE_COLORS.get(target.weakness, (255, 255, 255))
            pygame.draw.circle(self.screen, reticle_color, target.pos, target.radius + 5, 2)
            
    def _render_ui(self):
        # Render health bars for mechs and enemies
        for entity in self.mechs + self.enemies:
            entity.draw_health_bar(self.screen, self.COLOR_UI_BAR_BG)
        
        # Render cooldowns for mechs
        for mech in self.mechs:
            mech.draw_cooldown(self.screen)

        # Render Combo Multiplier
        if self.combo_multiplier > 1.0:
            size_multiplier = 1 + min((self.combo_multiplier - 1.0) * 0.1, 1.0)
            font = pygame.font.SysFont("Arial", int(48 * size_multiplier), bold=True)
            text = f"{self.combo_multiplier:.2f}x"
            text_surf = font.render(text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, 50))
            self.screen.blit(text_surf, text_rect)
        
        # Render Wave Info
        wave_text = f"WAVE {self.current_wave}/{self.total_waves}"
        text_surf = self.font_main.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))

        # Render Score
        current_score = self.score + int(self.reward_this_step)
        score_text = f"SCORE: {current_score}"
        text_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Render Mech Type Labels
        for i, mech in enumerate(self.mechs):
            label = f"{i+1}: {mech.type}"
            text_surf = self.font_small.render(label, True, mech.color)
            self.screen.blit(text_surf, (mech.pos.x - 20, mech.pos.y + 15))


    def _get_info(self):
        self.score += self.reward_this_step
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "combo_multiplier": self.combo_multiplier,
            "mechs_alive": sum(1 for m in self.mechs if m.is_alive()),
            "enemies_left": len(self.enemies),
        }

    def close(self):
        pygame.quit()

# --- Helper Classes ---

class Mech:
    def __init__(self, pos, type, color):
        self.pos = pos
        self.type = type
        self.color = color
        self.max_health = 100
        self.health = self.max_health
        self.radius = 12
        self.cooldown_duration = 30 # 1 second at 30fps
        self.cooldown_timer = 0

    def is_alive(self):
        return self.health > 0

    def update(self):
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

    def fire(self, target_pos):
        self.cooldown_timer = self.cooldown_duration
        direction = (target_pos - self.pos).normalize()
        return Projectile(self.pos.copy(), direction, self.type, self.color)

    def draw(self, surface, is_selected):
        if not self.is_alive(): return
        
        # Main body
        points = [
            self.pos + pygame.math.Vector2(0, -self.radius),
            self.pos + pygame.math.Vector2(self.radius * 0.866, self.radius * 0.5),
            self.pos + pygame.math.Vector2(-self.radius * 0.866, self.radius * 0.5),
        ]
        pygame.gfxdraw.aapolygon(surface, [(int(p.x), int(p.y)) for p in points], self.color)
        pygame.gfxdraw.filled_polygon(surface, [(int(p.x), int(p.y)) for p in points], self.color)

        if is_selected:
            pygame.draw.circle(surface, (255, 255, 255), self.pos, self.radius + 4, 2)

    def draw_health_bar(self, surface, bg_color):
        if not self.is_alive(): return
        bar_pos = self.pos + pygame.math.Vector2(-self.radius, -self.radius - 10)
        width, height = self.radius * 2, 5
        health_ratio = max(0, self.health / self.max_health)
        pygame.draw.rect(surface, bg_color, (bar_pos.x, bar_pos.y, width, height))
        pygame.draw.rect(surface, GameEnv.COLOR_PLAYER, (bar_pos.x, bar_pos.y, width * health_ratio, height))

    def draw_cooldown(self, surface):
        if self.cooldown_timer > 0:
            progress = self.cooldown_timer / self.cooldown_duration
            end_angle = -math.pi / 2 + (2 * math.pi * (1 - progress))
            pygame.draw.arc(surface, self.color, (self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2), -math.pi/2, end_angle, 3)

class Enemy:
    def __init__(self, pos, health, weakness, np_random):
        self.pos = pos
        self.max_health = health
        self.health = health
        self.weakness = weakness
        self.radius = 15
        self.move_timer = 0
        self.move_target = pos.copy()
        self.np_random = np_random

    def is_alive(self):
        return self.health > 0

    def take_damage(self, amount):
        self.health -= amount

    def update(self):
        if self.move_timer <= 0:
            self.move_timer = self.np_random.integers(30, 90)
            self.move_target = pygame.math.Vector2(
                self.np_random.uniform(GameEnv.SCREEN_WIDTH * 0.5, GameEnv.SCREEN_WIDTH * 0.95),
                self.np_random.uniform(25, GameEnv.SCREEN_HEIGHT - 25)
            )
        else:
            self.move_timer -= 1
        
        direction = self.move_target - self.pos
        if direction.length() > 1:
            self.pos += direction.normalize() * 0.5

    def draw(self, surface):
        if not self.is_alive(): return
        num_sides = 6
        points = []
        for i in range(num_sides):
            angle = 2 * math.pi * i / num_sides
            p = self.pos + pygame.math.Vector2(self.radius * math.cos(angle), self.radius * math.sin(angle))
            points.append((int(p.x), int(p.y)))
        
        pygame.gfxdraw.aapolygon(surface, points, GameEnv.COLOR_ENEMY)
        pygame.gfxdraw.filled_polygon(surface, points, GameEnv.COLOR_ENEMY)

    def draw_health_bar(self, surface, bg_color):
        if not self.is_alive(): return
        bar_pos = self.pos + pygame.math.Vector2(-self.radius, -self.radius - 10)
        width, height = self.radius * 2, 5
        health_ratio = max(0, self.health / self.max_health)
        pygame.draw.rect(surface, bg_color, (bar_pos.x, bar_pos.y, width, height))
        pygame.draw.rect(surface, GameEnv.COLOR_ENEMY, (bar_pos.x, bar_pos.y, width * health_ratio, height))

class Projectile:
    def __init__(self, pos, direction, type, color):
        self.pos = pos
        self.vel = direction * 12
        self.type = type
        self.color = color
        self.radius = 5
        self.damage = 20
        self.active = True
        self.trail = []

    def update(self):
        self.trail.append(self.pos.copy())
        if len(self.trail) > 5:
            self.trail.pop(0)
        self.pos += self.vel

    def draw(self, surface):
        # Draw trail
        if len(self.trail) > 1:
            for i, p in enumerate(self.trail):
                alpha = int(255 * (i / len(self.trail)))
                trail_color = self.color + (alpha,)
                pygame.gfxdraw.filled_circle(surface, int(p.x), int(p.y), int(self.radius * (i/len(self.trail))), trail_color)

        # Draw main projectile
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        
class Particle:
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pos
        self.vel = vel
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def is_alive(self):
        return self.lifespan > 0

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95 # friction
        self.lifespan -= 1

    def draw(self, surface):
        if not self.is_alive(): return
        
        alpha = int(255 * (self.lifespan / self.max_lifespan))
        current_radius = int(self.radius * (self.lifespan / self.max_lifespan))
        if current_radius > 0:
            color_with_alpha = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), current_radius, color_with_alpha)

    @staticmethod
    def create_explosion(pos, color, np_random):
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 5)
        vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
        radius = np_random.uniform(2, 6)
        lifespan = np_random.integers(15, 30)
        return Particle(pos.copy(), vel, radius, color, lifespan)

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Mech Command")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]: movement_action = 1
        elif keys[pygame.K_2]: movement_action = 2
        elif keys[pygame.K_3]: movement_action = 3
        elif keys[pygame.K_4]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Rendering for Human ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
        if done:
            print(f"Game Over! Final Score: {info['score']}. Total Reward: {total_reward}")

    env.close()