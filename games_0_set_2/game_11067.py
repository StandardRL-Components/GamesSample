import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:35:12.329653
# Source Brief: brief_01067.md
# Brief Index: 1067
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# A helper class for Mechs to keep the code organized
class Mech:
    def __init__(self, pos, is_player, mech_type='grunt'):
        self.pos = pygame.Vector2(pos)
        self.is_player = is_player
        self.mech_type = mech_type
        
        self.max_health = 100
        self.health = self.max_health
        self.radius = 12
        self.speed = 1.5
        self.range = 150
        self.fire_rate = 60 # steps per shot
        self.fire_cooldown = 0
        
        self.target_pos = pygame.Vector2(pos)
        self.target_entity = None
        self.is_alive = True

    def update(self, game_speed):
        if not self.is_alive:
            return

        # Movement
        move_vec = self.target_pos - self.pos
        if move_vec.length() > 1:
            self.pos += move_vec.normalize() * self.speed * game_speed

        # Cooldown
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1 * game_speed

    def set_move_target(self, pos):
        self.target_pos = pygame.Vector2(pos)

    def can_fire(self):
        return self.is_alive and self.fire_cooldown <= 0

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            # SFX: Mech Explosion
        return not self.is_alive

    def draw(self, screen, is_selected=False):
        if not self.is_alive:
            return

        color = (60, 120, 255) if self.is_player else (255, 60, 60)
        
        # Determine angle for mech orientation
        angle_rad = 0
        if (self.target_pos - self.pos).length() > 1:
            angle_rad = math.atan2(self.target_pos.y - self.pos.y, self.target_pos.x - self.pos.x)
        
        # Simple triangular shape for mech
        p1 = (self.pos.x + self.radius * math.cos(angle_rad), self.pos.y + self.radius * math.sin(angle_rad))
        p2 = (self.pos.x + self.radius * 0.7 * math.cos(angle_rad + 2.2), self.pos.y + self.radius * 0.7 * math.sin(angle_rad + 2.2))
        p3 = (self.pos.x + self.radius * 0.7 * math.cos(angle_rad - 2.2), self.pos.y + self.radius * 0.7 * math.sin(angle_rad - 2.2))
        body_points = [p1, p2, p3]

        # Draw glow effect
        for i in range(4, 0, -1):
            glow_color = (*color, 20 * i)
            pygame.gfxdraw.filled_polygon(screen, [(int(p[0]), int(p[1])) for p in body_points], glow_color)
            pygame.gfxdraw.aapolygon(screen, [(int(p[0]), int(p[1])) for p in body_points], glow_color)

        # Draw main body
        pygame.gfxdraw.filled_polygon(screen, [(int(p[0]), int(p[1])) for p in body_points], color)
        pygame.gfxdraw.aapolygon(screen, [(int(p[0]), int(p[1])) for p in body_points], (200, 200, 220))

        # Draw selection indicator
        if is_selected:
            pygame.gfxdraw.aacircle(screen, int(self.pos.x), int(self.pos.y), self.radius + 5, (255, 255, 0, 200))
            pygame.gfxdraw.aacircle(screen, int(self.pos.x), int(self.pos.y), self.radius + 6, (255, 255, 0, 150))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Command a squad of mechs in a real-time strategy battle. Destroy the enemy base while defending your own."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to select a mech. Press space to command the selected mech to attack the enemy base. Hold shift to activate time-slow."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("Consolas", 16)
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 30)


        # --- Game Constants ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 55)
        self.COLOR_PLAYER = (60, 120, 255)
        self.COLOR_ENEMY = (255, 60, 60)
        self.COLOR_HEALTH = (50, 220, 50)
        self.COLOR_TIME_CHARGE = (255, 220, 50)
        self.MAX_STEPS = 2000
        self.TIME_SLOW_MAX_CHARGE = 300
        self.TIME_SLOW_RECHARGE_RATE = 0.2
        self.TIME_SLOW_DEPLETE_RATE = 1.0
        self.ENEMY_AI_DELAY = 50
        
        # Will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_mechs = []
        self.enemy_mechs = []
        self.player_base = {}
        self.enemy_base = {}
        self.projectiles = []
        self.particles = []
        self.time_slow_charge = 0
        self.is_time_slow_active = False
        self.selected_mech_idx = 0
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_base = {'pos': pygame.Vector2(50, self.HEIGHT / 2), 'health': 500, 'max_health': 500, 'radius': 25}
        self.enemy_base = {'pos': pygame.Vector2(self.WIDTH - 50, self.HEIGHT / 2), 'health': 500, 'max_health': 500, 'radius': 25}
        
        self.player_mechs = [
            Mech((100, self.HEIGHT / 2 - 60), is_player=True),
            Mech((100, self.HEIGHT / 2), is_player=True),
            Mech((100, self.HEIGHT / 2 + 60), is_player=True),
        ]
        self.enemy_mechs = [
            Mech((self.WIDTH - 100, self.HEIGHT / 2 - 60), is_player=False),
            Mech((self.WIDTH - 100, self.HEIGHT / 2), is_player=False),
            Mech((self.WIDTH - 100, self.HEIGHT / 2 + 60), is_player=False),
        ]
        
        self.projectiles = []
        self.particles = []
        
        self.time_slow_charge = self.TIME_SLOW_MAX_CHARGE
        self.is_time_slow_active = False
        self.selected_mech_idx = 0
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        self._handle_actions(action)
        self._update_time_slow(action[2] == 1)
        
        game_speed = 0.3 if self.is_time_slow_active else 1.0

        if self.steps > self.ENEMY_AI_DELAY:
            self._update_ai(game_speed)
            
        for mech in self.player_mechs + self.enemy_mechs:
            mech.update(game_speed)

        reward += self._update_projectiles_and_firing(game_speed)
        self._update_particles()
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.score += reward

        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True
            truncated = True # Use truncated for time limit
        
        self.game_over = terminated or truncated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Mech Selection ---
        living_player_mechs = [m for m in self.player_mechs if m.is_alive]
        if not living_player_mechs:
            self.selected_mech_idx = -1
            return

        # Sort mechs by Y then X for predictable selection
        living_player_mechs.sort(key=lambda m: (m.pos.y, m.pos.x))
        
        current_selection_mech = self.player_mechs[self.selected_mech_idx]
        try:
            current_sorted_idx = living_player_mechs.index(current_selection_mech)
        except ValueError: # If selected mech died, re-select
            current_sorted_idx = 0

        if movement in [1, 2]: # Up/Down
            new_idx = (current_sorted_idx - 1) if movement == 1 else (current_sorted_idx + 1)
            new_idx %= len(living_player_mechs)
            selected_mech = living_player_mechs[new_idx]
            self.selected_mech_idx = self.player_mechs.index(selected_mech)

        elif movement in [3, 4]: # Left/Right
            # Sort by X then Y for left/right selection
            living_player_mechs.sort(key=lambda m: (m.pos.x, m.pos.y))
            try:
                current_sorted_idx = living_player_mechs.index(current_selection_mech)
            except ValueError:
                current_sorted_idx = 0
            
            new_idx = (current_sorted_idx - 1) if movement == 3 else (current_sorted_idx + 1)
            new_idx %= len(living_player_mechs)
            selected_mech = living_player_mechs[new_idx]
            self.selected_mech_idx = self.player_mechs.index(selected_mech)

        # --- Command Mech ---
        space_just_pressed = space_pressed and not self.prev_space_held
        if space_just_pressed and self.selected_mech_idx != -1:
            mech = self.player_mechs[self.selected_mech_idx]
            if mech.is_alive:
                # SFX: Command Acknowledged
                mech.set_move_target(self.enemy_base['pos'])
        
        self.prev_space_held = space_pressed

    def _update_time_slow(self, shift_held):
        if shift_held and self.time_slow_charge > 0:
            self.is_time_slow_active = True
            self.time_slow_charge -= self.TIME_SLOW_DEPLETE_RATE
            if self.time_slow_charge < 0: self.time_slow_charge = 0
        else:
            self.is_time_slow_active = False
            self.time_slow_charge += self.TIME_SLOW_RECHARGE_RATE
            if self.time_slow_charge > self.TIME_SLOW_MAX_CHARGE:
                self.time_slow_charge = self.TIME_SLOW_MAX_CHARGE

    def _update_ai(self, game_speed):
        living_player_mechs = [m for m in self.player_mechs if m.is_alive]
        if not living_player_mechs: return

        for enemy_mech in self.enemy_mechs:
            if not enemy_mech.is_alive: continue
            
            # Find closest player mech
            closest_player_mech = min(living_player_mechs, key=lambda m: m.pos.distance_to(enemy_mech.pos))
            
            # Set target to move towards
            enemy_mech.set_move_target(closest_player_mech.pos)

    def _update_projectiles_and_firing(self, game_speed):
        reward = 0
        
        # Firing logic
        all_mechs = self.player_mechs + self.enemy_mechs
        for mech in all_mechs:
            if not mech.is_alive or not mech.can_fire(): continue
            
            targets = self.enemy_mechs if mech.is_player else self.player_mechs
            target_base = self.enemy_base if mech.is_player else self.player_base

            # Prioritize mechs in range
            potential_targets = [t for t in targets if t.is_alive and mech.pos.distance_to(t.pos) < mech.range]
            
            if potential_targets:
                target = min(potential_targets, key=lambda t: t.pos.distance_to(mech.pos))
                self._create_projectile(mech, target.pos)
            # If no mechs in range, target base
            elif mech.pos.distance_to(target_base['pos']) < mech.range:
                self._create_projectile(mech, target_base['pos'])

        # Update projectiles
        for p in self.projectiles[:]:
            p['pos'] += p['vel'] * game_speed
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.projectiles.remove(p)
                continue
            
            # Collision check
            targets = self.enemy_mechs if p['is_player_owned'] else self.player_mechs
            target_base = self.enemy_base if p['is_player_owned'] else self.player_base
            
            hit = False
            for target_mech in targets:
                if target_mech.is_alive and p['pos'].distance_to(target_mech.pos) < target_mech.radius:
                    was_destroyed = target_mech.take_damage(p['damage'])
                    if p['is_player_owned']:
                        reward += 5.0 if was_destroyed else 0.1
                    else: # Player mech was hit
                        reward -= 5.0 if was_destroyed else 0.1
                    hit = True
                    break
            
            if not hit and p['pos'].distance_to(target_base['pos']) < target_base['radius']:
                target_base['health'] -= p['damage']
                if p['is_player_owned']:
                    reward += 0.2
                hit = True

            if hit:
                self._create_explosion(p['pos'], p['color'])
                self.projectiles.remove(p)
        
        return reward

    def _create_projectile(self, owner_mech, target_pos):
        # SFX: Laser Fire
        vel = (target_pos - owner_mech.pos).normalize() * 5
        color = self.COLOR_PLAYER if owner_mech.is_player else self.COLOR_ENEMY
        self.projectiles.append({
            'pos': owner_mech.pos.copy(),
            'vel': vel,
            'color': color,
            'is_player_owned': owner_mech.is_player,
            'damage': 20,
            'lifespan': 150
        })
        owner_mech.fire_cooldown = owner_mech.fire_rate

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1
            if p['lifespan'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, num_particles=20):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': random.uniform(2, 5),
                'color': (*color, random.randint(100, 255)),
                'lifespan': random.randint(20, 40)
            })

    def _check_termination(self):
        player_mechs_alive = any(m.is_alive for m in self.player_mechs)
        if not player_mechs_alive or self.player_base['health'] <= 0:
            return True, -100.0 # Loss
        
        if self.enemy_base['health'] <= 0:
            return True, 100.0 # Win
        
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw bases and their health bars
        self._draw_base(self.player_base, self.COLOR_PLAYER)
        self._draw_base(self.enemy_base, self.COLOR_ENEMY)

        # Draw projectiles and particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])
        for p in self.projectiles:
            start_pos = (int(p['pos'].x), int(p['pos'].y))
            end_pos = (int(p['pos'].x - p['vel'].x * 2), int(p['pos'].y - p['vel'].y * 2))
            pygame.draw.line(self.screen, (*p['color'], 100), start_pos, end_pos, 4)
            pygame.draw.line(self.screen, p['color'], start_pos, end_pos, 2)

        # Draw mechs and their health bars
        all_mechs = self.player_mechs + self.enemy_mechs
        all_mechs.sort(key=lambda m: m.pos.y) # Draw from top to bottom
        for i, mech in enumerate(all_mechs):
            is_selected = False
            if mech.is_player and self.selected_mech_idx >= 0 and self.player_mechs[self.selected_mech_idx].is_alive:
                 is_selected = self.player_mechs.index(mech) == self.selected_mech_idx
            mech.draw(self.screen, is_selected)
            if mech.is_alive:
                self._draw_health_bar(mech.pos - (0, mech.radius + 8), mech.health, mech.max_health)
        
        # Time slow overlay
        if self.is_time_slow_active:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 50, 100, 80))
            self.screen.blit(overlay, (0,0))

    def _draw_base(self, base, color):
        pos = (int(base['pos'].x), int(base['pos'].y))
        radius = int(base['radius'])
        
        # Glow
        for i in range(10, 0, -2):
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + i, (*color, 15 * i))
        
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (200, 200, 220))
        
        self._draw_health_bar(base['pos'] - (0, radius + 8), base['health'], base['max_health'], width=50)

    def _draw_health_bar(self, pos, health, max_health, width=30, height=5):
        if health < 0: health = 0
        ratio = health / max_health
        
        bg_rect = pygame.Rect(pos.x - width/2, pos.y - height/2, width, height)
        fg_rect = pygame.Rect(pos.x - width/2, pos.y - height/2, width * ratio, height)
        
        pygame.draw.rect(self.screen, (80, 20, 20), bg_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, fg_rect, border_radius=2)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, (220, 220, 240))
        self.screen.blit(score_text, (10, 10))

        # Time Slow Bar
        bar_width = 200
        bar_height = 15
        bar_x = self.WIDTH / 2 - bar_width / 2
        bar_y = self.HEIGHT - 25
        
        charge_ratio = self.time_slow_charge / self.TIME_SLOW_MAX_CHARGE
        
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        fg_rect = pygame.Rect(bar_x, bar_y, bar_width * charge_ratio, bar_height)
        
        pygame.draw.rect(self.screen, (50, 50, 50), bg_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_TIME_CHARGE, fg_rect, border_radius=3)
        
        label_text = self.font_small.render("TIME-SLOW", True, (220, 220, 240))
        text_rect = label_text.get_rect(center=bg_rect.center)
        self.screen.blit(label_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_mechs_alive": sum(1 for m in self.player_mechs if m.is_alive),
            "enemy_mechs_alive": sum(1 for m in self.enemy_mechs if m.is_alive),
            "enemy_base_health": self.enemy_base['health'],
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Control Setup ---
    # Re-initialize pygame with a visible display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Mech Command")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0
    shift_held = 0

    print("\n--- Manual Control ---")
    print(env.user_guide)
    print("R: Reset Environment")
    print("Q: Quit")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: running = False
                if event.key == pygame.K_r: 
                    obs, info = env.reset()
                    total_reward = 0
                
                # Update action state on key down
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1

            if event.type == pygame.KEYUP:
                # Reset non-held keys
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0
                if event.key == pygame.K_SPACE:
                    space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0
        
        # Assemble action from key states
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # For manual play, we need to reset movement after one step
        movement = 0

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()