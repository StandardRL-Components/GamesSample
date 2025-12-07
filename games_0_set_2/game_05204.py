
# Generated: 2025-08-28T04:17:52.197637
# Source Brief: brief_05204.md
# Brief Index: 5204

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move. Hold shift for a powerful special attack. Press space to perform a quick basic attack."
    )

    game_description = (
        "Defeat waves of monsters and a final boss in this fast-paced, side-view 2D fighter. Time your attacks and dodges to survive."
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
        
        # --- Constants & Config ---
        self.GROUND_Y = self.HEIGHT - 70
        self.MAX_STEPS = 2500
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_SPEED = 5
        self.PLAYER_INVULN_FRAMES = 60 # 2 seconds
        self.BASIC_ATTACK_COOLDOWN = 15 # 0.5 seconds
        self.SPECIAL_ATTACK_COOLDOWN = 120 # 4 seconds
        self.WAVES_BEFORE_BOSS = 5

        # --- Colors ---
        self.COLOR_BG = (18, 23, 33)
        self.COLOR_GROUND = (48, 38, 51)
        self.COLOR_PLAYER = (60, 180, 255)
        self.COLOR_PLAYER_HURT = (255, 100, 100)
        self.COLOR_ENEMY_SLIME = (130, 220, 70)
        self.COLOR_ENEMY_BAT = (180, 80, 200)
        self.COLOR_ENEMY_MAGE = (240, 150, 50)
        self.COLOR_BOSS = (255, 60, 90)
        self.COLOR_HEALTH_GREEN = (40, 200, 80)
        self.COLOR_HEALTH_RED = (220, 50, 50)
        self.COLOR_HEALTH_BG = (40, 40, 40)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_DMG_TEXT = (255, 220, 100)
        
        # --- Fonts ---
        self.font_ui = pygame.font.Font(None, 24)
        self.font_dmg = pygame.font.Font(None, 22)
        self.font_wave = pygame.font.Font(None, 48)

        # --- State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        
        self.player = {}
        self.enemies = []
        self.player_attacks = []
        self.enemy_projectiles = []
        self.particles = []
        self.damage_texts = []
        
        self.wave = 0
        self.boss_spawned = False
        self.boss_defeated = False
        self.special_cooldown_timer = 0
        self.basic_cooldown_timer = 0
        self.screen_shake = 0
        self.wave_transition_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        
        self.player = {
            'rect': pygame.Rect(100, self.GROUND_Y - 50, 30, 50),
            'health': self.PLAYER_MAX_HEALTH,
            'vel_x': 0,
            'state': 'idle',
            'state_timer': 0,
            'facing_right': True,
            'invuln_timer': 0
        }
        
        self.enemies.clear()
        self.player_attacks.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        self.damage_texts.clear()
        
        self.wave = 0
        self.boss_spawned = False
        self.boss_defeated = False
        self.special_cooldown_timer = 0
        self.basic_cooldown_timer = 0
        self.screen_shake = 0
        self.wave_transition_timer = 120 # Show "Wave 1" message

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = 0
        self.steps += 1
        
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
        else:
            self._handle_input(action)
            self._update_player()
            self._update_enemies()
            self._update_attacks()
            self._update_projectiles()
        
        self._update_effects()
        self._handle_collisions()
        
        if not self.boss_spawned and not self.enemies and self.wave_transition_timer <= 0 and not self.boss_defeated:
            self.wave_transition_timer = 120
            self._spawn_wave()
            
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        can_act = self.player['state'] in ['idle', 'run']

        # Movement
        if movement == 3: # Left
            self.player['vel_x'] = -self.PLAYER_SPEED
            self.player['facing_right'] = False
        elif movement == 4: # Right
            self.player['vel_x'] = self.PLAYER_SPEED
            self.player['facing_right'] = True
        else:
            self.player['vel_x'] = 0
        
        # Actions
        if can_act:
            if shift_pressed and self.special_cooldown_timer <= 0:
                # sfx: special_attack_charge.wav
                self.player['state'] = 'special'
                self.player['state_timer'] = 30 # 1s duration
                self.special_cooldown_timer = self.SPECIAL_ATTACK_COOLDOWN
                self.screen_shake = 10
                # Spawn a large, persistent attack hitbox
                attack_rect = pygame.Rect(self.player['rect'].centerx - 150, self.GROUND_Y - 100, 300, 100)
                self.player_attacks.append({'rect': attack_rect, 'type': 'special', 'lifetime': 25, 'damage': 25})
            elif space_pressed and self.basic_cooldown_timer <= 0:
                # sfx: sword_swing.wav
                self.player['state'] = 'attack'
                self.player['state_timer'] = 15 # 0.5s duration
                self.basic_cooldown_timer = self.BASIC_ATTACK_COOLDOWN
                # Spawn a short-lived attack hitbox
                x_offset = 20 if self.player['facing_right'] else -70
                attack_rect = pygame.Rect(self.player['rect'].x + x_offset, self.player['rect'].y, 50, 50)
                self.player_attacks.append({'rect': attack_rect, 'type': 'basic', 'lifetime': 8, 'damage': 10})

    def _update_player(self):
        p = self.player
        # Update timers
        if p['state_timer'] > 0: p['state_timer'] -= 1
        else: p['state'] = 'idle'
        if p['invuln_timer'] > 0: p['invuln_timer'] -= 1
        if self.special_cooldown_timer > 0: self.special_cooldown_timer -= 1
        if self.basic_cooldown_timer > 0: self.basic_cooldown_timer -= 1

        # Apply velocity
        p['rect'].x += p['vel_x']
        
        # Boundary checks
        p['rect'].left = max(0, p['rect'].left)
        p['rect'].right = min(self.WIDTH, p['rect'].right)

    def _update_enemies(self):
        for e in self.enemies:
            e['state_timer'] = max(0, e['state_timer'] - 1)
            
            # Basic AI: move towards player
            player_cx = self.player['rect'].centerx
            dist_x = player_cx - e['rect'].centerx
            
            if e['type'] == 'slime':
                if e['state'] == 'idle' and e['state_timer'] == 0:
                    e['state'] = 'jump'
                    e['state_timer'] = self.np_random.integers(30, 60)
                    e['vel_y'] = -8
                    e['vel_x'] = 4 * np.sign(dist_x) if abs(dist_x) > 20 else 0
                if e['state'] == 'jump':
                    e['rect'].x += e['vel_x']
                    e['rect'].y += e['vel_y']
                    e['vel_y'] += 0.5 # Gravity
                    if e['rect'].bottom >= self.GROUND_Y:
                        e['rect'].bottom = self.GROUND_Y
                        e['state'] = 'idle'
                        e['state_timer'] = self.np_random.integers(30, 90)
            
            elif e['type'] == 'bat':
                e['sine_angle'] = (e['sine_angle'] + 0.1) % (2 * math.pi)
                e['rect'].y = e['start_y'] + math.sin(e['sine_angle']) * 30
                e['rect'].x += 2.5 * np.sign(dist_x)

            elif e['type'] == 'mage' or e['type'] == 'boss':
                desired_dist = 250 if e['type'] == 'mage' else 300
                if abs(dist_x) > desired_dist:
                    e['rect'].x += 1.5 * np.sign(dist_x)
                if e['state'] == 'idle' and e['state_timer'] == 0:
                    e['state'] = 'casting'
                    e['state_timer'] = 60 if e['type'] == 'mage' else 45
                if e['state'] == 'casting' and e['state_timer'] == 15:
                    # sfx: magic_shoot.wav
                    proj_speed = 6 if e['type'] == 'mage' else 8
                    proj_size = (12, 12) if e['type'] == 'mage' else (20, 20)
                    self.enemy_projectiles.append({
                        'rect': pygame.Rect(e['rect'].centerx - proj_size[0]/2, e['rect'].centery - proj_size[1]/2, *proj_size),
                        'vel_x': proj_speed * np.sign(dist_x),
                        'damage': 15 if e['type'] == 'mage' else 25
                    })
                if e['state_timer'] == 0:
                    e['state'] = 'idle'
                    e['state_timer'] = self.np_random.integers(90, 150)
            
            e['rect'].left = max(0, e['rect'].left)
            e['rect'].right = min(self.WIDTH, e['rect'].right)

    def _update_attacks(self):
        self.player_attacks = [a for a in self.player_attacks if a['lifetime'] > 0]
        for a in self.player_attacks:
            a['lifetime'] -= 1

    def _update_projectiles(self):
        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p['rect'].centerx < self.WIDTH]
        for p in self.enemy_projectiles:
            p['rect'].x += p['vel_x']

    def _update_effects(self):
        # Particles
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['lifetime'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
        
        # Damage Texts
        self.damage_texts = [d for d in self.damage_texts if d['lifetime'] > 0]
        for d in self.damage_texts:
            d['lifetime'] -= 1
            d['pos'][1] -= 0.5
        
        # Screen Shake
        if self.screen_shake > 0: self.screen_shake -= 1

    def _handle_collisions(self):
        # Player attacks vs Enemies
        for attack in self.player_attacks:
            if 'hit_enemies' not in attack:
                attack['hit_enemies'] = []
            for i, enemy in enumerate(self.enemies):
                if i not in attack['hit_enemies'] and attack['rect'].colliderect(enemy['rect']):
                    # sfx: hit_confirm.wav
                    damage = attack['damage']
                    enemy['health'] -= damage
                    self.reward_this_step += 0.1
                    self.score += damage
                    attack['hit_enemies'].append(i)
                    self._spawn_damage_text(enemy['rect'].center, damage)
                    self._spawn_particles(enemy['rect'].center, 10, (255, 255, 100), 4, 3)

        # Check for dead enemies
        alive_enemies = []
        for enemy in self.enemies:
            if enemy['health'] > 0:
                alive_enemies.append(enemy)
            else:
                # sfx: enemy_die.wav
                if enemy['type'] == 'boss':
                    self.reward_this_step += 100
                    self.score += 1000
                    self.boss_defeated = True
                    self._spawn_particles(enemy['rect'].center, 100, (255, 200, 50), 8, 5)
                else:
                    self.reward_this_step += 1
                    self.score += 100
                    self._spawn_particles(enemy['rect'].center, 30, (150, 150, 150), 5, 4)
        self.enemies = alive_enemies

        # Enemy attacks vs Player
        if self.player['invuln_timer'] > 0: return

        # Contact damage
        for enemy in self.enemies:
            if self.player['rect'].colliderect(enemy['rect']):
                self._damage_player(10)
                return

        # Projectile damage
        for proj in self.enemy_projectiles:
            if self.player['rect'].colliderect(proj['rect']):
                self._damage_player(proj['damage'])
                proj['rect'].x = -1000 # Mark for removal
                return
    
    def _damage_player(self, amount):
        # sfx: player_hurt.wav
        self.player['health'] -= amount
        self.player['invuln_timer'] = self.PLAYER_INVULN_FRAMES
        self.player['state'] = 'hurt'
        self.player['state_timer'] = 20
        self.reward_this_step -= 0.2
        self.screen_shake = 15
        self._spawn_particles(self.player['rect'].center, 15, self.COLOR_PLAYER_HURT, 4, 3)
        self._spawn_damage_text(self.player['rect'].center, amount)

    def _spawn_wave(self):
        self.wave += 1
        if self.wave > self.WAVES_BEFORE_BOSS and not self.boss_spawned:
            # Spawn Boss
            self.boss_spawned = True
            self.enemies.append({
                'rect': pygame.Rect(self.WIDTH - 150, self.GROUND_Y - 100, 80, 100),
                'health': 200, 'max_health': 200, 'type': 'boss', 'state': 'idle',
                'state_timer': 60
            })
        elif not self.boss_spawned:
            # Spawn regular wave
            num_enemies = self.wave
            for _ in range(num_enemies):
                enemy_type = self.np_random.choice(['slime', 'bat', 'mage'])
                x = self.np_random.integers(self.WIDTH // 2, self.WIDTH - 50)
                if enemy_type == 'slime':
                    self.enemies.append({
                        'rect': pygame.Rect(x, self.GROUND_Y - 40, 40, 40),
                        'health': 30, 'max_health': 30, 'type': 'slime', 'state': 'idle',
                        'state_timer': 0, 'vel_x': 0, 'vel_y': 0
                    })
                elif enemy_type == 'bat':
                    y = self.np_random.integers(100, self.GROUND_Y - 100)
                    self.enemies.append({
                        'rect': pygame.Rect(x, y, 30, 20),
                        'health': 20, 'max_health': 20, 'type': 'bat', 'start_y': y,
                        'sine_angle': self.np_random.uniform(0, 2*math.pi), 'state_timer': 0
                    })
                elif enemy_type == 'mage':
                     self.enemies.append({
                        'rect': pygame.Rect(x, self.GROUND_Y - 60, 35, 60),
                        'health': 25, 'max_health': 25, 'type': 'mage', 'state': 'idle',
                        'state_timer': self.np_random.integers(30, 120)
                    })

    def _check_termination(self):
        if self.player['health'] <= 0: return True
        if self.boss_defeated: return True
        if self.steps >= self.MAX_STEPS: return True
        return False

    def _get_observation(self):
        render_offset = [0, 0]
        if self.screen_shake > 0:
            render_offset[0] = self.np_random.integers(-5, 5)
            render_offset[1] = self.np_random.integers(-5, 5)

        self.screen.fill(self.COLOR_BG)
        self._render_background(render_offset)
        self._render_entities(render_offset)
        self._render_effects(render_offset)
        self._render_ui()

        if self.wave_transition_timer > 0:
            self._render_wave_text()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self, offset):
        # Ground
        ground_rect = pygame.Rect(0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y)
        ground_rect.move_ip(offset)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)
        # Some distant "mountains" for parallax
        for i in range(5):
            x = (i * 200 - (self.steps * 0.1) % 200) + offset[0]
            pygame.gfxdraw.aatrigon(self.screen, 
                                    int(x), int(self.GROUND_Y + offset[1]), 
                                    int(x + 100), int(self.GROUND_Y - 150 + offset[1]),
                                    int(x + 200), int(self.GROUND_Y + offset[1]),
                                    (40, 35, 45))
    
    def _render_entities(self, offset):
        # Player
        p_color = self.COLOR_PLAYER
        if self.player['invuln_timer'] > 0 and self.steps % 10 < 5:
            p_color = self.COLOR_PLAYER_HURT
        p_rect = self.player['rect'].move(offset)
        pygame.draw.rect(self.screen, p_color, p_rect, border_radius=3)
        self._render_health_bar(p_rect.centerx, p_rect.top - 10, self.player['health'], self.PLAYER_MAX_HEALTH, self.COLOR_HEALTH_GREEN)

        # Enemies
        for e in self.enemies:
            e_rect = e['rect'].move(offset)
            color = self.COLOR_ENEMY_SLIME
            if e['type'] == 'bat': color = self.COLOR_ENEMY_BAT
            elif e['type'] == 'mage': color = self.COLOR_ENEMY_MAGE
            elif e['type'] == 'boss': color = self.COLOR_BOSS
            
            if e['type'] == 'slime':
                squash = 1.0
                if e['state'] == 'jump':
                    squash = 1.0 + e['vel_y'] * 0.05
                h = max(5, e_rect.height * squash)
                w = max(5, e_rect.width / max(0.5, squash))
                squashed_rect = pygame.Rect(e_rect.centerx - w/2, e_rect.bottom - h, w, h)
                pygame.draw.ellipse(self.screen, color, squashed_rect)
            else:
                pygame.draw.rect(self.screen, color, e_rect, border_radius=3)

            self._render_health_bar(e_rect.centerx, e_rect.top - 10, e['health'], e['max_health'], self.COLOR_HEALTH_RED)

    def _render_effects(self, offset):
        # Player Attacks
        for a in self.player_attacks:
            a_rect = a['rect'].move(offset)
            if a['type'] == 'basic':
                alpha = int(255 * (a['lifetime'] / 8))
                s = pygame.Surface(a_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, (255, 255, 255, alpha), s.get_rect(), border_radius=5)
                self.screen.blit(s, a_rect.topleft)
            elif a['type'] == 'special':
                alpha = int(255 * math.sin((a['lifetime'] / 25) * math.pi))
                s = pygame.Surface(a_rect.size, pygame.SRCALPHA)
                pygame.gfxdraw.box(s, s.get_rect(), (255, 255, 255, alpha//2))
                pygame.gfxdraw.rectangle(s, s.get_rect(), (255, 255, 255, alpha))
                self.screen.blit(s, a_rect.topleft)

        # Enemy Projectiles
        for p in self.enemy_projectiles:
            p_rect = p['rect'].move(offset)
            pygame.draw.ellipse(self.screen, self.COLOR_BOSS, p_rect)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['size']), color)
        
        # Damage Texts
        for d in self.damage_texts:
            alpha = int(255 * (d['lifetime'] / d['max_lifetime']))
            text_surf = self.font_dmg.render(str(d['amount']), True, self.COLOR_DMG_TEXT)
            text_surf.set_alpha(alpha)
            pos = (int(d['pos'][0] + offset[0]), int(d['pos'][1] + offset[1]))
            self.screen.blit(text_surf, text_surf.get_rect(center=pos))

    def _render_health_bar(self, x, y, health, max_health, color):
        width, height = 40, 5
        bg_rect = pygame.Rect(x - width/2, y, width, height)
        health_ratio = max(0, health / max_health)
        fill_rect = pygame.Rect(x - width/2, y, width * health_ratio, height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect)
        pygame.draw.rect(self.screen, color, fill_rect)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Special Cooldown
        bar_w, bar_h = 100, 15
        x, y = 10, 35
        cooldown_ratio = 1.0 - (self.special_cooldown_timer / self.SPECIAL_ATTACK_COOLDOWN)
        text_surf = self.font_ui.render("SPECIAL", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (x, y))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (x + 70, y, bar_w, bar_h))
        if cooldown_ratio >= 1.0:
            pygame.draw.rect(self.screen, (255, 220, 0), (x + 70, y, bar_w, bar_h))
        else:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (x + 70, y, bar_w * cooldown_ratio, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (x + 70, y, bar_w, bar_h), 1)

    def _render_wave_text(self):
        alpha = 255
        if self.wave_transition_timer < 30: alpha = int(255 * (self.wave_transition_timer / 30))
        if self.wave_transition_timer > 90: alpha = int(255 * ((120 - self.wave_transition_timer) / 30))
        
        text = f"WAVE {self.wave}"
        if self.boss_spawned: text = "FINAL BOSS"
        
        wave_surf = self.font_wave.render(text, True, self.COLOR_UI_TEXT)
        wave_surf.set_alpha(alpha)
        self.screen.blit(wave_surf, wave_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 50)))

    def _spawn_particles(self, pos, count, color, speed, size):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5), 
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)]
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'max_lifetime': lifetime, 'color': color, 'size': size})

    def _spawn_damage_text(self, pos, amount):
        lifetime = 45
        self.damage_texts.append({'pos': list(pos), 'amount': amount, 'lifetime': lifetime, 'max_lifetime': lifetime})
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player['health'],
            "wave": self.wave,
            "boss_defeated": self.boss_defeated
        }
    
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()