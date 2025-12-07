import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:32:53.026448
# Source Brief: brief_02235.md
# Brief Index: 2235
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro arcade game.
    The player defends their arcade cabinet by firing puns at incoming enemies.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Defend your retro arcade cabinet from waves of enemies by firing an arsenal of powerful puns."
    )
    user_guide = (
        "Controls: Use ↑↓ to select your pun and ←→ to select your item. "
        "Press space to fire and shift to use the selected item."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.ARCADE_MAX_HEALTH = 100

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (150, 255, 200)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_BOSS = (255, 150, 50)
        self.COLOR_PROJECTILE_PLAYER = (255, 255, 0)
        self.COLOR_PROJECTILE_ENEMY = (255, 100, 200)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (50, 60, 80, 180)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 50, 50)
        self.COLOR_ITEM_SHIELD = (0, 150, 255)
        self.COLOR_ITEM_AMP = (255, 100, 0)

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.arcade_health = 0
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.player_fire_cooldown = 0
        self.enemy_spawn_timer = 0
        self.difficulty_mod = 1.0
        self.spawn_rate_mod = 1.0
        self.boss_spawn_steps = [2000, 4000]
        self.boss_active = False
        
        # Pun and Item mechanics
        self.puns = [
            ("BEAR", "NECESSITIES", 15), ("CLAWS", "AND EFFECT", 20),
            ("A-SALT", "RIFLE", 12), ("HOLY", "CARP", 10),
            ("LETTUCE", "PRAY", 18), ("TIME", "FLIES", 25)
        ]
        self.selected_pun_idx = 0
        self.inventory = {"SHIELD": 1, "AMPLIFIER": 1}
        self.item_list = list(self.inventory.keys())
        self.selected_item_idx = 0
        self.item_shield_timer = 0
        self.item_amp_active = False
        self.item_use_cooldown = 0

        # Action state tracking for rising edge detection
        self.prev_space_held = False
        self.prev_shift_held = False

        # Initialize state by calling reset
        # self.reset() # reset() is called by the environment wrapper
        
        # Run validation check
        # self.validate_implementation() # No need to run this in production


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.arcade_health = self.ARCADE_MAX_HEALTH
        
        self.enemies.clear()
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        self.player_fire_cooldown = 0
        self.enemy_spawn_timer = 60  # Initial delay
        self.difficulty_mod = 1.0
        self.spawn_rate_mod = 1.0
        self.boss_active = False
        
        self.selected_pun_idx = 0
        self.inventory = {"SHIELD": 1, "AMPLIFIER": 1}
        self.selected_item_idx = 0
        self.item_shield_timer = 0
        self.item_amp_active = False
        self.item_use_cooldown = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        # Spawn initial enemy
        self._spawn_enemy()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        
        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Detect rising edge for presses
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        # Update action state trackers
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Cooldowns tick down
        if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
        if self.item_use_cooldown > 0: self.item_use_cooldown -= 1
        if self.item_shield_timer > 0: self.item_shield_timer -= 1
        
        # Action: Select Pun (Up/Down)
        if movement == 1: # Up
            self.selected_pun_idx = (self.selected_pun_idx - 1) % len(self.puns)
        elif movement == 2: # Down
            self.selected_pun_idx = (self.selected_pun_idx + 1) % len(self.puns)
            
        # Action: Select Item (Left/Right)
        if movement == 3: # Left
            self.selected_item_idx = (self.selected_item_idx - 1) % len(self.item_list)
        elif movement == 4: # Right
            self.selected_item_idx = (self.selected_item_idx + 1) % len(self.item_list)

        # Action: Fire Pun (Space)
        if space_pressed and self.player_fire_cooldown == 0:
            # sfx: player_shoot.wav
            self.player_fire_cooldown = 20 # 20 frames cooldown
            _, _, damage = self.puns[self.selected_pun_idx]
            if self.item_amp_active:
                damage *= 2
                self.item_amp_active = False
            self.player_projectiles.append({
                "pos": np.array([120, 200], dtype=float), "vel": np.array([10, 0], dtype=float),
                "damage": damage, "text": self.puns[self.selected_pun_idx][0]
            })

        # Action: Use Item (Shift)
        if shift_pressed and self.item_use_cooldown == 0:
            item_name = self.item_list[self.selected_item_idx]
            if self.inventory[item_name] > 0:
                # sfx: powerup.wav
                self.inventory[item_name] -= 1
                self.item_use_cooldown = 30 # 1s cooldown
                if item_name == "SHIELD":
                    self.item_shield_timer = 150 # 5 seconds of shield
                elif item_name == "AMPLIFIER":
                    self.item_amp_active = True

        # --- 2. Update Game State ---
        self.steps += 1
        
        # Update difficulty
        if self.steps % 500 == 0:
            self.difficulty_mod += 0.1
            self.spawn_rate_mod = max(0.5, self.spawn_rate_mod - 0.05)
            
        # Update player projectiles
        for p in self.player_projectiles[:]:
            p['pos'] += p['vel']
            if p['pos'][0] > self.WIDTH:
                self.player_projectiles.remove(p)

        # Update enemy projectiles
        for p in self.enemy_projectiles[:]:
            p['pos'] += p['vel']
            if p['pos'][0] < 100: # Hit the arcade
                self.enemy_projectiles.remove(p)
                if self.item_shield_timer <= 0:
                    # sfx: arcade_hit.wav
                    self.arcade_health -= p['damage']
                    reward -= 0.1
                    self._create_particles(np.array([100, p['pos'][1]]), 20, self.COLOR_ENEMY)
                else:
                    # sfx: shield_block.wav
                    self._create_particles(np.array([100, p['pos'][1]]), 20, self.COLOR_ITEM_SHIELD)

        # Update enemies
        for e in self.enemies[:]:
            e['pos'] += e['vel']
            e['attack_cooldown'] -= 1
            if e['attack_cooldown'] <= 0:
                # sfx: enemy_shoot.wav
                self.enemy_projectiles.append({
                    "pos": e['pos'].copy(), "vel": np.array([-5, 0], dtype=float),
                    "damage": e['damage'], "text": "ARGH"
                })
                e['attack_cooldown'] = e['attack_rate']
            if e['pos'][0] < 80: # Reached the arcade
                self.enemies.remove(e)
                self.arcade_health -= 25 # High penalty for letting them through
                reward -= 1.0

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # --- 3. Handle Collisions ---
        for pp in self.player_projectiles[:]:
            for e in self.enemies[:]:
                if np.linalg.norm(pp['pos'] - e['pos']) < e['size']:
                    # sfx: enemy_hit.wav
                    self.player_projectiles.remove(pp)
                    e['health'] -= pp['damage']
                    self.score += int(pp['damage'])
                    reward += 0.1
                    self._create_particles(e['pos'], 15, self.COLOR_PROJECTILE_PLAYER)
                    if e['health'] <= 0:
                        # sfx: enemy_explode.wav
                        self._create_particles(e['pos'], 50, e['color'])
                        self.score += e['score_value']
                        reward += 5.0 if e['is_boss'] else 1.0
                        if e['is_boss']:
                            self.boss_active = False
                            if self.steps >= self.boss_spawn_steps[-1]: # Final boss
                                terminated = True
                                reward += 100
                                self.game_over = True
                        self.enemies.remove(e)
                    break # Projectile is consumed

        # --- 4. Spawn New Enemies ---
        self.enemy_spawn_timer -= 1
        if self.steps in self.boss_spawn_steps and not self.boss_active:
            self._spawn_enemy(is_boss=True)
            self.boss_active = True
        elif self.enemy_spawn_timer <= 0 and not self.boss_active:
            self._spawn_enemy()
            self.enemy_spawn_timer = int((120 + self.np_random.integers(-20, 20)) * self.spawn_rate_mod)
        
        # --- 5. Check Termination ---
        if self.arcade_health <= 0:
            self.arcade_health = 0
            terminated = True
            reward -= 100
            self.game_over = True
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Game ends on step limit
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_enemy(self, is_boss=False):
        y_pos = self.np_random.uniform(50, self.HEIGHT - 50)
        if is_boss:
            self.enemies.append({
                "pos": np.array([self.WIDTH + 50, y_pos], dtype=float),
                "vel": np.array([-0.5, 0], dtype=float),
                "health": 500 * self.difficulty_mod,
                "max_health": 500 * self.difficulty_mod,
                "damage": 20,
                "attack_cooldown": 60,
                "attack_rate": 60,
                "size": 40,
                "color": self.COLOR_ENEMY_BOSS,
                "score_value": 100,
                "is_boss": True
            })
        else:
            self.enemies.append({
                "pos": np.array([self.WIDTH + 20, y_pos], dtype=float),
                "vel": np.array([-1.5, 0], dtype=float),
                "health": 30 * self.difficulty_mod,
                "max_health": 30 * self.difficulty_mod,
                "damage": 10,
                "attack_cooldown": 100,
                "attack_rate": int(self.np_random.uniform(120, 180) * self.spawn_rate_mod),
                "size": 15,
                "color": self.COLOR_ENEMY,
                "score_value": 10,
                "is_boss": False
            })

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "color": color,
                "lifespan": self.np_random.integers(15, 30)
            })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "arcade_health": self.arcade_health}

    def _get_observation(self):
        # --- Render Background ---
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # --- Render Game Objects ---
        # Arcade Cabinet
        pygame.draw.rect(self.screen, (50, 50, 60), (0, 0, 100, self.HEIGHT))
        pygame.draw.rect(self.screen, (30, 30, 40), (10, 10, 80, self.HEIGHT - 20))
        
        # Player Pun Cannon
        cannon_pos = (100, 200)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, cannon_pos, 20)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_GLOW, cannon_pos, 22, 2)
        if self.player_fire_cooldown > 15: # Charging effect
            charge_radius = (20 - self.player_fire_cooldown) * 4
            pygame.gfxdraw.aacircle(self.screen, cannon_pos[0], cannon_pos[1], max(0, charge_radius), self.COLOR_PLAYER_GLOW)

        # Shield Effect
        if self.item_shield_timer > 0:
            alpha = 100 + (self.item_shield_timer % 10) * 5
            shield_surf = pygame.Surface((20, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(shield_surf, (*self.COLOR_ITEM_SHIELD, alpha), (0, 0, 20, self.HEIGHT))
            self.screen.blit(shield_surf, (90, 0))

        # Particles
        for p in self.particles:
            alpha = max(0, p['lifespan'] * 8)
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

        # Projectiles
        for p in self.player_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE_PLAYER, (int(p['pos'][0]), int(p['pos'][1])), 5)
        for p in self.enemy_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE_ENEMY, (int(p['pos'][0]), int(p['pos'][1])), 4)

        # Enemies
        for e in self.enemies:
            pos = (int(e['pos'][0]), int(e['pos'][1]))
            size = int(e['size'])
            pygame.draw.circle(self.screen, e['color'], pos, size)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, size, 2)
            # Health bar
            health_ratio = max(0, e['health'] / e['max_health'])
            pygame.draw.rect(self.screen, (50,0,0), (pos[0] - size, pos[1] - size - 10, size*2, 5))
            pygame.draw.rect(self.screen, (0,200,0), (pos[0] - size, pos[1] - size - 10, int(size*2*health_ratio), 5))

        # --- Render UI ---
        # Arcade Health Bar
        health_ratio = max(0, self.arcade_health / self.ARCADE_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (120, 10, self.WIDTH - 130, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (120, 10, (self.WIDTH - 130) * health_ratio, 20))
        health_text = self.font_small.render(f"ARCADE HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (125, 12))
        
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 35))

        # Pun Selection UI
        pun_ui_surf = pygame.Surface((200, 180), pygame.SRCALPHA)
        pun_ui_surf.fill(self.COLOR_UI_BG)
        for i, pun in enumerate(self.puns):
            color = self.COLOR_PLAYER if i == self.selected_pun_idx else self.COLOR_UI_TEXT
            text = f"{pun[0]} {pun[1]}"
            pun_text = self.font_small.render(text, True, color)
            pun_ui_surf.blit(pun_text, (10, 10 + i * 25))
            if i == self.selected_pun_idx:
                 pygame.draw.rect(pun_ui_surf, self.COLOR_PLAYER, (2, 8 + i * 25, 4, 20), 0)
        self.screen.blit(pun_ui_surf, (10, self.HEIGHT - 190))

        # Item Selection UI
        for i, item_name in enumerate(self.item_list):
            is_selected = i == self.selected_item_idx
            box_rect = pygame.Rect(120 + i * 70, self.HEIGHT - 60, 60, 50)
            bg_color = self.COLOR_ITEM_SHIELD if item_name == "SHIELD" else self.COLOR_ITEM_AMP
            pygame.draw.rect(self.screen, (*bg_color, 90), box_rect)
            if is_selected:
                pygame.draw.rect(self.screen, (255,255,255), box_rect, 3)
            
            item_text = self.font_small.render(item_name[:4], True, self.COLOR_UI_TEXT)
            self.screen.blit(item_text, (box_rect.centerx - item_text.get_width()//2, box_rect.y + 5))
            
            count_text = self.font_medium.render(f"x{self.inventory[item_name]}", True, self.COLOR_UI_TEXT)
            self.screen.blit(count_text, (box_rect.centerx - count_text.get_width()//2, box_rect.y + 25))

        # Amplifier active indicator
        if self.item_amp_active:
            amp_text = self.font_medium.render("AMP ON!", True, self.COLOR_ITEM_AMP)
            pygame.draw.circle(self.screen, self.COLOR_ITEM_AMP, (cannon_pos[0], cannon_pos[1]), 25, 3)
            self.screen.blit(amp_text, (cannon_pos[0] + 30, cannon_pos[1] - 15))
        
        # Game Over Screen
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            win = self.arcade_health > 0
            end_text_str = "YOU WIN!" if win else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_PLAYER if win else self.COLOR_ENEMY)
            self.screen.blit(end_text, (self.WIDTH//2 - end_text.get_width()//2, self.HEIGHT//2 - end_text.get_height()//2 - 20))
            final_score_text = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            self.screen.blit(final_score_text, (self.WIDTH//2 - final_score_text.get_width()//2, self.HEIGHT//2 + 20))

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def render(self):
        # This method is not used by the agent but can be used for human playing
        # Note: The environment is designed for rgb_array mode, so this is just for display
        obs = self._get_observation()
        # Pygame uses (width, height), numpy uses (height, width)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # If a display surface doesn't exist, create one
        if not getattr(self, 'display_screen', None):
            pygame.display.init()
            self.display_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        
        self.display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # --- Human Controls ---
    # W/S or Up/Down: Select Pun
    # A/D or Left/Right: Select Item
    # Space: Fire
    # Left Shift: Use Item
    # Q: Quit
    
    while not done:
        # Default action is NO-OP
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: action[0] = 1 # Up
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: action[0] = 2 # Down
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: action[0] = 3 # Left
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: action[0] = 4 # Right
        if keys[pygame.K_SPACE]: action[1] = 1 # Space
        if keys[pygame.K_LSHIFT]: action[2] = 1 # Shift
        if keys[pygame.K_q]: done = True
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause on game over screen
            obs, info = env.reset()

    env.close()