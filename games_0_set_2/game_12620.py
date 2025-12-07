import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:56:03.607501
# Source Brief: brief_02620.md
# Brief Index: 2620
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Functions and Classes ---

def draw_text(surface, text, position, font, color, anchor="center"):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    setattr(text_rect, anchor, position)
    surface.blit(text_surface, text_rect)

def draw_polygon_aa(surface, points, color, width=1):
    pygame.draw.aalines(surface, color, True, points, 1)
    if width > 1:
        pygame.draw.polygon(surface, color, points)

def draw_glowing_circle(surface, center, radius, color, glow_color):
    for i in range(int(radius * 1.5), int(radius), -1):
        alpha = int(100 * (1 - (i - radius) / (radius * 0.5)))
        if alpha > 0:
            rgb_glow = glow_color[:3]
            pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), i, (*rgb_glow, alpha))
    pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), color)
    pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), color)

class Particle:
    def __init__(self, x, y, color, size, life, angle=None, speed=None):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        if angle is None:
            angle = random.uniform(0, 2 * math.pi)
        if speed is None:
            speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.vx *= 0.98
        self.vy *= 0.98

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            current_size = int(self.size * (self.life / self.max_life))
            if current_size > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((current_size * 2, current_size * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, current_size, current_size, current_size, (*self.color, alpha))
                surface.blit(temp_surf, (int(self.x) - current_size, int(self.y) - current_size))


class Projectile:
    def __init__(self, x, y, target_x, target_y, color, speed, damage, is_enemy):
        self.x = x
        self.y = y
        self.color = color
        self.speed = speed
        self.damage = damage
        self.is_enemy = is_enemy
        angle = math.atan2(target_y - y, target_x - x)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = 120 # 4 seconds at 30fps

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        end_x = self.x - self.vx * 0.5
        end_y = self.y - self.vy * 0.5
        pygame.draw.aaline(surface, self.color, (self.x, self.y), (end_x, end_y), 3)

class Ship:
    def __init__(self, x, y, health):
        self.x = x
        self.y = y
        self.health = health
        self.max_health = health
        self.size = 12
        self.fire_cooldown = 0

    def update(self, env):
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

    def draw(self, surface):
        # Health bar
        if self.health < self.max_health:
            bar_width = 20
            bar_height = 3
            health_pct = self.health / self.max_health
            fill_width = int(bar_width * health_pct)
            pygame.draw.rect(surface, (255, 0, 0), (self.x - bar_width / 2, self.y - self.size - 8, bar_width, bar_height))
            pygame.draw.rect(surface, (0, 255, 0), (self.x - bar_width / 2, self.y - self.size - 8, fill_width, bar_height))
    
    def is_dead(self):
        return self.health <= 0

class PlayerShip(Ship):
    def __init__(self, x, y, health=20):
        super().__init__(x, y, health)
        self.size = 15

class AttackShip(PlayerShip):
    def update(self, env):
        super().update(env)
        if self.fire_cooldown == 0 and env.enemy_ships:
            # Target closest enemy
            closest_enemy = min(env.enemy_ships, key=lambda e: math.hypot(e.x - self.x, e.y - self.y))
            env.projectiles.append(Projectile(self.x, self.y, closest_enemy.x, closest_enemy.y, env.COLOR_PLAYER, 8, 5, False))
            self.fire_cooldown = 60 # Fire every 2 seconds
            # sfx: player_laser.wav

    def draw(self, surface):
        points = [
            (self.x, self.y - self.size),
            (self.x - self.size / 2, self.y + self.size / 2),
            (self.x + self.size / 2, self.y + self.size / 2),
        ]
        draw_polygon_aa(surface, points, GameEnv.COLOR_PLAYER, width=2)
        super().draw(surface)

class SupportShip(PlayerShip):
    def __init__(self, x, y):
        super().__init__(x, y, health=30)
        self.size = 14
    
    def update(self, env):
        super().update(env)
        if self.fire_cooldown == 0:
            if env.player_health < env.max_player_health:
                env.player_health = min(env.max_player_health, env.player_health + 1)
                # sfx: heal.wav
                for _ in range(5):
                    env.particles.append(Particle(self.x, self.y, GameEnv.COLOR_SUPPORT, random.randint(2, 4), 30))
            self.fire_cooldown = 90 # Heal every 3 seconds

    def draw(self, surface):
        draw_glowing_circle(surface, (self.x, self.y), self.size * 0.7, GameEnv.COLOR_SUPPORT, (0, 255, 0))
        super().draw(surface)

class TargetingShip(PlayerShip):
    def __init__(self, x, y):
        super().__init__(x, y, health=15)
        self.size = 13

    def update(self, env):
        super().update(env) # Does nothing but cool down

    def draw(self, surface):
        points = [
            (self.x, self.y - self.size),
            (self.x + self.size, self.y),
            (self.x, self.y + self.size),
            (self.x - self.size, self.y),
        ]
        draw_polygon_aa(surface, points, GameEnv.COLOR_STRIKE, width=2)
        super().draw(surface)

class EnemyShip(Ship):
    def __init__(self, x, y, health, speed, target_y):
        super().__init__(x, y, health)
        self.speed = speed
        self.target_y = target_y
        self.size = 14
        self.fire_cooldown = random.randint(90, 180)

    def update(self, env):
        super().update(env)
        # Move to position
        if self.y < self.target_y:
            self.y += self.speed
        
        # Fire at player fleet
        if self.fire_cooldown == 0:
            target_x = random.randint(200, 440)
            target_y = 380
            env.projectiles.append(Projectile(self.x, self.y, target_x, target_y, env.COLOR_ENEMY, 5, 5, True))
            self.fire_cooldown = 120 + random.randint(-30, 30) # Fire every 4 seconds
            # sfx: enemy_laser.wav

    def draw(self, surface):
        points = [
            (self.x - self.size, self.y - self.size / 2),
            (self.x + self.size, self.y - self.size / 2),
            (self.x, self.y + self.size),
        ]
        draw_polygon_aa(surface, points, GameEnv.COLOR_ENEMY, width=2)
        super().draw(surface)

# --- Main Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your fleet by strategically placing attack, support, and targeting ships to fend off waves of alien invaders."
    )
    user_guide = (
        "Use ←→ arrow keys to select a ship card, press Space to choose. "
        "Use arrow keys to position the ship, press Space again to deploy. "
        "Press Shift to activate a special attack with two targeting ships."
    )
    auto_advance = True
    
    # Color Palette
    COLOR_BG = (10, 20, 40)
    COLOR_STAR = (100, 120, 150)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_SUPPORT = (50, 255, 100)
    COLOR_STRIKE = (255, 200, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_PANEL = (20, 40, 70, 200)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        self.font_s = pygame.font.Font(None, 20)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 48)
        
        self.starfield = [(random.randint(0, self.width), random.randint(0, self.height), random.randint(1, 2)) for _ in range(150)]
        
        self.max_steps = 2000
        self.total_waves = 10
        
        # Initialize state variables
        self._initialize_state()

    def _initialize_state(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.reward_buffer = 0
        
        self.player_health = 100
        self.max_player_health = 100
        
        self.wave_number = 0
        self.game_state = 'SELECT_CARD' # or 'PLACE_SHIP'
        
        self.deck_composition = ['ATTACK', 'ATTACK', 'ATTACK', 'SUPPORT', 'TARGET']
        self.max_hand_size = 4
        self.cards_in_hand = []
        self.selected_card_index = 0
        self.card_to_place = None
        
        self.placement_cursor = [self.width / 2, self.height / 2 + 50]
        
        self.player_ships = []
        self.enemy_ships = []
        self.projectiles = []
        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False
        self.last_move_action = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        
        self.wave_number = 1
        self._spawn_wave()
        
        while len(self.cards_in_hand) < self.max_hand_size:
            self._draw_card()
            
        return self._get_observation(), self._get_info()

    def _spawn_wave(self):
        self.enemy_ships.clear()
        enemy_count = 3 + (self.wave_number - 1) // 2
        enemy_health = 10 + (self.wave_number - 1) // 3
        
        for i in range(enemy_count):
            x = self.width * (i + 1) / (enemy_count + 1)
            y = -20
            target_y = random.randint(50, 150)
            speed = random.uniform(0.5, 1.0)
            self.enemy_ships.append(EnemyShip(x, y, enemy_health, speed, target_y))
        # sfx: new_wave.wav
    
    def _draw_card(self):
        if len(self.cards_in_hand) < self.max_hand_size:
            self.cards_in_hand.append(random.choice(self.deck_composition))

    def step(self, action):
        self.reward_buffer = 0
        self.game_over = self.player_health <= 0 or self.steps >= self.max_steps
        if self.game_over:
            self.reward_buffer += -100 if self.player_health <= 0 else 0

        if not self.game_over and not self.win:
            self._handle_input(action)
            self._update_game_logic()

        terminated = self.game_over or self.win
        truncated = self.steps >= self.max_steps
        
        self.steps += 1
        self.score += self.reward_buffer
        
        return (
            self._get_observation(),
            self.reward_buffer,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held

        # Debounce movement for menu navigation
        move_press = movement != 0 and movement != self.last_move_action
        
        if self.game_state == 'SELECT_CARD':
            if move_press:
                if movement == 3: # Left
                    self.selected_card_index = max(0, self.selected_card_index - 1)
                elif movement == 4: # Right
                    self.selected_card_index = min(len(self.cards_in_hand) - 1, self.selected_card_index + 1)
            
            if space_press and self.cards_in_hand:
                self.card_to_place = self.cards_in_hand[self.selected_card_index]
                self.game_state = 'PLACE_SHIP'
                # sfx: menu_select.wav

        elif self.game_state == 'PLACE_SHIP':
            cursor_speed = 5
            if movement == 1: self.placement_cursor[1] -= cursor_speed
            if movement == 2: self.placement_cursor[1] += cursor_speed
            if movement == 3: self.placement_cursor[0] -= cursor_speed
            if movement == 4: self.placement_cursor[0] += cursor_speed
            
            self.placement_cursor[0] = np.clip(self.placement_cursor[0], 20, self.width - 20)
            self.placement_cursor[1] = np.clip(self.placement_cursor[1], 200, self.height - 80)

            if space_press:
                self._deploy_ship()
                self.game_state = 'SELECT_CARD'
                self.selected_card_index = min(self.selected_card_index, len(self.cards_in_hand) - 1)
        
        if shift_press:
            self._try_combo()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        self.last_move_action = movement if movement != 0 else self.last_move_action

    def _deploy_ship(self):
        x, y = self.placement_cursor
        ship_type = self.card_to_place
        
        if ship_type == 'ATTACK':
            self.player_ships.append(AttackShip(x, y))
        elif ship_type == 'SUPPORT':
            self.player_ships.append(SupportShip(x, y))
        elif ship_type == 'TARGET':
            self.player_ships.append(TargetingShip(x, y))
        
        self.cards_in_hand.pop(self.selected_card_index)
        self._draw_card()
        self.card_to_place = None
        # sfx: deploy_ship.wav

    def _try_combo(self):
        targeting_ships = [s for s in self.player_ships if isinstance(s, TargetingShip)]
        if len(targeting_ships) >= 2:
            self.reward_buffer += 1.0
            # sfx: orbital_strike_charge.wav
            
            # Damage all enemies
            for enemy in self.enemy_ships:
                enemy.health -= 25
                self.reward_buffer += 0.1
                # Create visual effect
                for _ in range(10):
                    self.particles.append(Particle(enemy.x, enemy.y, self.COLOR_STRIKE, random.randint(3, 8), 60))
            
            # Consume the two oldest targeting ships
            targeting_ships.sort(key=lambda s: s.fire_cooldown, reverse=True)
            for ship in targeting_ships[:2]:
                self.player_ships.remove(ship)

    def _update_game_logic(self):
        # Update entities
        for ship in self.player_ships: ship.update(self)
        for ship in self.enemy_ships: ship.update(self)
        for p in self.projectiles: p.update()
        for part in self.particles: part.update()
        
        # Projectile collisions
        for p in self.projectiles[:]:
            if p.is_enemy:
                # Check against player ships
                for ship in self.player_ships[:]:
                    if math.hypot(p.x - ship.x, p.y - ship.y) < ship.size:
                        ship.health -= p.damage
                        if p in self.projectiles: self.projectiles.remove(p)
                        break
                # Check against player base area
                if p.y > self.height - 70:
                    self.player_health -= p.damage
                    self.reward_buffer -= 0.1
                    if p in self.projectiles: self.projectiles.remove(p)
                    # sfx: player_hit.wav
            else: # Player projectile
                for ship in self.enemy_ships[:]:
                    if math.hypot(p.x - ship.x, p.y - ship.y) < ship.size:
                        ship.health -= p.damage
                        self.reward_buffer += 0.1
                        if p in self.projectiles: self.projectiles.remove(p)
                        break
        
        # Cleanup dead ships and expired items
        for ship in self.player_ships[:]:
            if ship.is_dead():
                self.reward_buffer -= 0.1
                self.player_ships.remove(ship)
                # sfx: explosion.wav
                for _ in range(30): self.particles.append(Particle(ship.x, ship.y, self.COLOR_PLAYER, random.randint(1, 4), 40))

        for ship in self.enemy_ships[:]:
            if ship.is_dead():
                self.enemy_ships.remove(ship)
                self.reward_buffer += 0.5
                # sfx: explosion.wav
                for _ in range(30): self.particles.append(Particle(ship.x, ship.y, self.COLOR_ENEMY, random.randint(1, 4), 40))

        self.projectiles = [p for p in self.projectiles if p.life > 0 and 0 < p.x < self.width and 0 < p.y < self.height]
        self.particles = [p for p in self.particles if p.life > 0]

        # Wave progression
        if not self.enemy_ships and not self.game_over:
            if self.wave_number >= self.total_waves:
                self.win = True
                self.reward_buffer += 100
            else:
                self.wave_number += 1
                self.reward_buffer += 2.0
                self._spawn_wave()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Starfield
        for x, y, size in self.starfield:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))
            
        # Entities
        for p in self.projectiles: p.draw(self.screen)
        for ship in self.player_ships: ship.draw(self.screen)
        for ship in self.enemy_ships: ship.draw(self.screen)
        for part in self.particles: part.draw(self.screen)

    def _render_ui(self):
        # Bottom UI Panel
        panel_rect = pygame.Rect(0, self.height - 60, self.width, 60)
        s = pygame.Surface((self.width, 60), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_PANEL)
        self.screen.blit(s, (0, self.height - 60))

        # Cards
        card_width, card_height = 80, 50
        total_card_width = len(self.cards_in_hand) * (card_width + 10) - 10
        start_x = (self.width - total_card_width) / 2
        
        for i, card_type in enumerate(self.cards_in_hand):
            card_x = start_x + i * (card_width + 10)
            card_y = self.height - 55
            card_rect = pygame.Rect(card_x, card_y, card_width, card_height)
            
            color_map = {'ATTACK': self.COLOR_PLAYER, 'SUPPORT': self.COLOR_SUPPORT, 'TARGET': self.COLOR_STRIKE}
            card_color = color_map.get(card_type, (100, 100, 100))
            
            if i == self.selected_card_index and self.game_state == 'SELECT_CARD':
                pygame.draw.rect(self.screen, (255, 255, 255), card_rect.inflate(6, 6), 2, 5)
            
            pygame.draw.rect(self.screen, card_color, card_rect, 0, 5)
            draw_text(self.screen, card_type, card_rect.center, self.font_s, (0,0,0))
            
        # Placement cursor
        if self.game_state == 'PLACE_SHIP':
            cursor_pos = (int(self.placement_cursor[0]), int(self.placement_cursor[1]))
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, 20, 20, 20, (255, 255, 255, 150))
            self.screen.blit(temp_surf, (cursor_pos[0] - 20, cursor_pos[1] - 20))
            
            pygame.draw.line(self.screen, (255, 255, 255, 150), (cursor_pos[0] - 10, cursor_pos[1]), (cursor_pos[0] + 10, cursor_pos[1]))
            pygame.draw.line(self.screen, (255, 255, 255, 150), (cursor_pos[0], cursor_pos[1] - 10), (cursor_pos[0], cursor_pos[1] + 10))

        # HUD Text
        draw_text(self.screen, f"Wave: {self.wave_number}/{self.total_waves}", (10, 10), self.font_m, self.COLOR_UI_TEXT, "topleft")
        draw_text(self.screen, f"Enemies: {len(self.enemy_ships)}", (self.width - 10, 10), self.font_m, self.COLOR_ENEMY, "topright")
        
        # Player Health
        draw_text(self.screen, "Fleet Integrity", (self.width - 10, self.height - 50), self.font_m, self.COLOR_UI_TEXT, "topright")
        health_bar_rect = pygame.Rect(self.width - 160, self.height - 25, 150, 15)
        health_pct = max(0, self.player_health / self.max_player_health)
        fill_rect = pygame.Rect(health_bar_rect.x, health_bar_rect.y, int(health_bar_rect.width * health_pct), health_bar_rect.height)
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, health_bar_rect, 0, 3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, fill_rect, 0, 3)
        
        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            draw_text(self.screen, "FLEET DESTROYED", (self.width/2, self.height/2), self.font_l, self.COLOR_ENEMY)
        elif self.win:
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            draw_text(self.screen, "VICTORY", (self.width/2, self.height/2), self.font_l, self.COLOR_STRIKE)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "player_health": self.player_health,
            "enemies_remaining": len(self.enemy_ships),
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Fleet Command")
    clock = pygame.time.Clock()

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Keyboard input to action mapping
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        # Cap the frame rate
        clock.tick(30)
        
    env.close()