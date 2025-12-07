import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:23:54.658485
# Source Brief: brief_03163.md
# Brief Index: 3163
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes to organize state
class Player:
    """Represents the player character."""
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.size = 20
        self.health = 100
        self.max_health = 100
        self.on_ground = False
        self.last_move_dir = 1  # 1 for right, -1 for left

class Boss:
    """Represents the enemy boss."""
    def __init__(self, pos, level):
        self.pos = pygame.math.Vector2(pos)
        self.size = 60
        self.max_health = 100 + (level - 1) * 25
        self.health = self.max_health
        self.attack_cooldown = 0
        self.attack_rate = 1.0 + (level - 1) * 0.1  # attacks per second
        self.move_direction = 1
        self.move_speed = 1.5

class Projectile:
    """Represents a magical projectile."""
    def __init__(self, pos, vel, owner, color):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.owner = owner  # 'player' or 'boss'
        self.color = color
        self.size = 8
        self.lifespan = 180  # 6 seconds at 30fps

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, pos, vel, color, size, lifespan):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.color = color
        self.size = size
        self.lifespan = lifespan

class Card:
    """Represents a magical card the player can use."""
    def __init__(self, name, description, cooldown):
        self.name = name
        self.description = description
        self.cooldown = cooldown

# --- Main Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Battle a powerful boss using a deck of magical cards. Jump on platforms, "
        "dodge projectiles, and unlock new abilities to win."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move, ↑ to jump, and ↓ to fall faster. "
        "Press Shift to cycle through your cards and Space to use the selected card."
    )
    auto_advance = True
    
    # Class-level state for persistent unlocks across episodes
    unlocked_card_names = ["Summon Platform", "Magic Missile"]
    current_level = 1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- Colors ---
        self.COLOR_BG = (15, 10, 30)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (50, 200, 255)
        self.COLOR_BOSS = (255, 60, 0)
        self.COLOR_BOSS_GLOW = (255, 120, 50)
        self.COLOR_PLATFORM = (0, 100, 200)
        self.COLOR_PLATFORM_GLOW = (50, 150, 220)
        self.COLOR_PLAYER_PROJ = (200, 220, 255)
        self.COLOR_BOSS_PROJ = (255, 100, 0)
        self.COLOR_TELEPORT = (180, 0, 255)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_BAR = (0, 200, 80)
        self.COLOR_HEALTH_BAR_BG = (80, 80, 80)
        
        # --- Game Parameters ---
        self.max_steps = 2000
        self.gravity = 0.4
        self.player_speed = 4.0
        self.player_jump_strength = -9.0
        
        # --- State Variables ---
        self.player = None
        self.boss = None
        self.platforms = []
        self.temp_platforms = []
        self.projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.card_cooldown = 0
        self.gravity_modifier = 1.0
        self.gravity_shift_timer = 0
        
        self.available_cards = [
            Card("Summon Platform", "Creates a temporary platform.", 60),
            Card("Magic Missile", "Fires a projectile.", 20),
            Card("Gravity Shift", "Reverses gravity for 3s.", 180),
            Card("Teleport", "Blinks forward.", 90)
        ]
        self.unlocked_cards = []
        self.current_card_index = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = Player(pos=(self.screen_width / 2, self.screen_height - 50))
        self.boss = Boss(pos=(self.screen_width / 2, 80), level=self.current_level)
        
        self.platforms = [pygame.Rect(0, self.screen_height - 20, self.screen_width, 20)]
        self.temp_platforms = []
        self.projectiles = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.card_cooldown = 0
        self.gravity_modifier = 1.0
        self.gravity_shift_timer = 0
        
        self.unlocked_cards = [card for card in self.available_cards if card.name in self.unlocked_card_names]
        self.current_card_index = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self._handle_player_movement(movement)
        
        if shift_pressed and self.unlocked_cards:
            self.current_card_index = (self.current_card_index + 1) % len(self.unlocked_cards)
            # SFX: Card cycle sound
            
        if space_pressed and self.card_cooldown <= 0 and self.unlocked_cards:
            card = self.unlocked_cards[self.current_card_index]
            card_used = self._use_card(card)
            if card_used:
                self.card_cooldown = card.cooldown
                reward += 1.0

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        self._update_player()
        self._update_boss()
        self._update_projectiles()
        self._update_particles()
        self._update_card_cooldowns()
        
        reward += self._handle_collisions()
        
        terminated = False
        if self.player.health <= 0:
            terminated = True
            reward -= 100
            self.game_over = True
        elif self.boss.health <= 0:
            terminated = True
            reward += 100
            self.game_over = True
            self._on_level_complete()
        elif self.steps >= self.max_steps:
            terminated = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 3: # Left
            self.player.vel.x = -self.player_speed
            self.player.last_move_dir = -1
        elif movement == 4: # Right
            self.player.vel.x = self.player_speed
            self.player.last_move_dir = 1
        else:
            self.player.vel.x = 0
            
        if movement == 1 and self.player.on_ground: # Up (Jump)
            self.player.vel.y = self.player_jump_strength * self.gravity_modifier
            self.player.on_ground = False
            # SFX: Jump sound
            self._spawn_particles(self.player.pos + pygame.math.Vector2(0, self.player.size/2), 10, self.COLOR_PLAYER_GLOW)
        
        if movement == 2 and not self.player.on_ground: # Down (Fast Fall)
            self.player.vel.y = max(self.player.vel.y, 3.0)

    def _update_player(self):
        self.player.vel.y += self.gravity * self.gravity_modifier
        self.player.pos += self.player.vel
        
        self.player.pos.x = np.clip(self.player.pos.x, self.player.size/2, self.screen_width - self.player.size/2)
        if self.player.pos.y > self.screen_height + self.player.size or self.player.pos.y < -self.player.size:
            self.player.health = 0

        player_rect = pygame.Rect(self.player.pos.x - self.player.size/2, self.player.pos.y - self.player.size/2, self.player.size, self.player.size)
        self.player.on_ground = False
        
        all_platform_rects = self.platforms + [p['rect'] for p in self.temp_platforms]
        for plat in all_platform_rects:
            if plat.colliderect(player_rect):
                if self.player.vel.y * self.gravity_modifier > 0: # Moving down
                    if player_rect.bottom > plat.top and player_rect.top < plat.top:
                        player_rect.bottom = plat.top
                        self.player.pos.y = player_rect.centery
                        self.player.vel.y = 0
                        self.player.on_ground = True
                elif self.player.vel.y * self.gravity_modifier < 0: # Moving up
                    if player_rect.top < plat.bottom and player_rect.bottom > plat.bottom:
                        player_rect.top = plat.bottom
                        self.player.pos.y = player_rect.centery
                        self.player.vel.y = 0
        
    def _update_boss(self):
        self.boss.pos.x += self.boss.move_speed * self.boss.move_direction
        if self.boss.pos.x < self.boss.size or self.boss.pos.x > self.screen_width - self.boss.size:
            self.boss.move_direction *= -1
            self.boss.pos.x = np.clip(self.boss.pos.x, self.boss.size, self.screen_width - self.boss.size)

        self.boss.attack_cooldown -= 1
        if self.boss.attack_cooldown <= 0:
            self.boss.attack_cooldown = (30 / self.boss.attack_rate) + random.uniform(-10, 10) # 30fps assumption
            target_vector = self.player.pos - self.boss.pos
            if target_vector.length() > 0:
                direction = target_vector.normalize()
                proj_vel = direction * 5
                self.projectiles.append(Projectile(self.boss.pos, proj_vel, 'boss', self.COLOR_BOSS_PROJ))
                # SFX: Boss attack sound
                self._spawn_particles(self.boss.pos, 15, self.COLOR_BOSS_GLOW)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj.pos += proj.vel
            proj.lifespan -= 1
            if proj.lifespan <= 0 or not (0 < proj.pos.x < self.screen_width and 0 < proj.pos.y < self.screen_height):
                self.projectiles.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p.pos += p.vel
            p.lifespan -= 1
            p.size = max(0, p.size - 0.2)
            if p.lifespan <= 0 or p.size <= 0:
                self.particles.remove(p)

    def _update_card_cooldowns(self):
        if self.card_cooldown > 0:
            self.card_cooldown -= 1
        
        if self.gravity_shift_timer > 0:
            self.gravity_shift_timer -= 1
            if self.gravity_shift_timer <= 0:
                self.gravity_modifier = 1.0

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player.pos.x - self.player.size/2, self.player.pos.y - self.player.size/2, self.player.size, self.player.size)
        boss_rect = pygame.Rect(self.boss.pos.x - self.boss.size/2, self.boss.pos.y - self.boss.size/2, self.boss.size, self.boss.size)
        
        for proj in self.projectiles[:]:
            proj_rect = pygame.Rect(proj.pos.x - proj.size/2, proj.pos.y - proj.size/2, proj.size, proj.size)
            if proj.owner == 'player' and proj_rect.colliderect(boss_rect):
                self.boss.health -= 10
                reward += 0.1
                self.projectiles.remove(proj)
                self._spawn_particles(proj.pos, 20, self.COLOR_PLAYER_PROJ)
                # SFX: Impact hit sound
            elif proj.owner == 'boss' and proj_rect.colliderect(player_rect):
                self.player.health -= 10
                reward -= 0.1
                self.projectiles.remove(proj)
                self._spawn_particles(proj.pos, 20, self.COLOR_BOSS_PROJ)
                # SFX: Player damage sound
        return reward

    def _use_card(self, card):
        if card.name == "Summon Platform":
            platform_pos = self.player.pos + pygame.math.Vector2(self.player.last_move_dir * 80, -40)
            platform_pos.x = np.clip(platform_pos.x, 50, self.screen_width - 50)
            platform_pos.y = np.clip(platform_pos.y, 50, self.screen_height - 50)
            new_plat_rect = pygame.Rect(int(platform_pos.x - 50), int(platform_pos.y - 5), 100, 10)
            self.temp_platforms.append({'rect': new_plat_rect, 'lifespan': 150}) # 5 seconds at 30fps
            # SFX: Platform summon
            self._spawn_particles(platform_pos, 30, self.COLOR_PLATFORM)
            return True
            
        elif card.name == "Magic Missile":
            direction = pygame.math.Vector2(self.player.last_move_dir, -0.2).normalize()
            proj_vel = direction * 8
            self.projectiles.append(Projectile(self.player.pos, proj_vel, 'player', self.COLOR_PLAYER_PROJ))
            # SFX: Magic missile cast
            return True
            
        elif card.name == "Gravity Shift":
            self.gravity_modifier *= -1
            self.gravity_shift_timer = 90 # 3 seconds at 30fps
            # SFX: Gravity shift activate
            self._spawn_particles(self.player.pos, 50, self.COLOR_TELEPORT)
            return True
            
        elif card.name == "Teleport":
            self.player.pos.x += self.player.last_move_dir * 150
            self.player.pos.x = np.clip(self.player.pos.x, self.player.size/2, self.screen_width - self.player.size/2)
            # SFX: Teleport whoosh
            self._spawn_particles(self.player.pos, 40, self.COLOR_TELEPORT)
            return True
        return False

    def _on_level_complete(self):
        GameEnv.current_level += 1
        potential_unlocks = [card.name for card in self.available_cards if card.name not in self.unlocked_card_names]
        if potential_unlocks:
            new_card = random.choice(potential_unlocks)
            self.unlocked_card_names.append(new_card)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render permanent platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            pygame.gfxdraw.rectangle(self.screen, plat, self.COLOR_PLATFORM_GLOW)
        
        # Render and update temporary platforms
        for plat_info in self.temp_platforms[:]:
            plat_rect = plat_info['rect']
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat_rect)
            pygame.gfxdraw.rectangle(self.screen, plat_rect, self.COLOR_PLATFORM_GLOW)
            plat_info['lifespan'] -= 1
            if plat_info['lifespan'] <= 0:
                self.temp_platforms.remove(plat_info)

        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.size), p.color)
        
        for proj in self.projectiles:
            p1, p2 = proj.pos, proj.pos - proj.vel.normalize() * 15
            pygame.draw.line(self.screen, (255,255,255), (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 4)
            pygame.draw.line(self.screen, proj.color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 3)

        boss_pos = (int(self.boss.pos.x), int(self.boss.pos.y))
        boss_size = int(self.boss.size / 2)
        for i in range(boss_size // 2, 0, -2):
            pygame.gfxdraw.filled_circle(self.screen, boss_pos[0], boss_pos[1], boss_size + i, (*self.COLOR_BOSS_GLOW, 100 - (i * 4)))
        pygame.gfxdraw.filled_circle(self.screen, boss_pos[0], boss_pos[1], boss_size, self.COLOR_BOSS)
        pygame.gfxdraw.aacircle(self.screen, boss_pos[0], boss_pos[1], boss_size, self.COLOR_BOSS_GLOW)
        
        player_pos = (int(self.player.pos.x), int(self.player.pos.y))
        player_size = int(self.player.size / 2)
        for i in range(player_size, 0, -2):
            pygame.gfxdraw.filled_circle(self.screen, player_pos[0], player_pos[1], player_size + i, (*self.COLOR_PLAYER_GLOW, 80 - (i * 5)))
        pygame.gfxdraw.filled_circle(self.screen, player_pos[0], player_pos[1], player_size, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos[0], player_pos[1], player_size, self.COLOR_PLAYER_GLOW)
        
        if self.gravity_shift_timer > 0:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            color = (*self.COLOR_TELEPORT, min(255, self.gravity_shift_timer * 3) // 4)
            s.fill(color)
            self.screen.blit(s, (0,0))

    def _render_ui(self):
        hp_text = self.font_medium.render(f"HP: {self.player.health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(hp_text, (10, 10))

        bar_w, bar_h = 300, 20
        health_pct = max(0, self.boss.health / self.boss.max_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.screen_width/2 - bar_w/2, 10, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (self.screen_width/2 - bar_w/2, 10, int(bar_w * health_pct), bar_h))
        
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 10, 10))

        if self.unlocked_cards:
            card = self.unlocked_cards[self.current_card_index]
            card_text = self.font_medium.render(card.name, True, self.COLOR_UI_TEXT)
            self.screen.blit(card_text, (self.screen_width - card_text.get_width() - 10, self.screen_height - 40))
            if self.card_cooldown > 0:
                cooldown_pct = self.card_cooldown / card.cooldown
                pygame.draw.rect(self.screen, (255,255,255,100), (self.screen_width - card_text.get_width() - 10, self.screen_height - 10, int(card_text.get_width() * cooldown_pct), 5))

        if self.game_over:
            result_text_str = "VICTORY" if self.boss.health <= 0 else "DEFEAT"
            result_color = (100, 255, 100) if self.boss.health <= 0 else (255, 100, 100)
            result_text = self.font_large.render(result_text_str, True, result_color)
            text_rect = result_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(result_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player.health,
            "boss_health": self.boss.health,
            "level": self.current_level,
            "unlocked_cards": [c.name for c in self.unlocked_cards]
        }

    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = random.uniform(2, 6)
            lifespan = random.randint(20, 40)
            self.particles.append(Particle(pos, vel, color, size, lifespan))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Card Gauntlet")
    clock = pygame.time.Clock()
    
    running = True
    
    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Level: {info['level']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            
        clock.tick(30)
        
    env.close()