
# Generated: 2025-08-27T18:29:28.193146
# Source Brief: brief_01848.md
# Brief Index: 1848

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to attack."
    )

    # Short, user-facing description of the game
    game_description = (
        "Defeat waves of monsters in a fast-paced, side-scrolling pixel-art world."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.W, self.H = 640, 400
        self.LEVEL_WIDTH = self.W * 3
        self.GROUND_Y = self.H - 50
        self.MAX_STEPS = 1000
        self.MONSTERS_PER_WAVE = 15

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GROUND = (48, 38, 52)
        self.COLOR_PLAYER = (66, 231, 255)
        self.COLOR_MONSTER = (255, 89, 114)
        self.COLOR_PLAYER_ATTACK = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_HEALTH = (48, 224, 112)
        self.COLOR_UI_HEALTH_BG = (192, 40, 60)
        self.COLOR_PARALLAX_1 = (25, 28, 42)
        self.COLOR_PARALLAX_2 = (35, 38, 52)

        # Physics
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 6
        self.JUMP_STRENGTH = 15
        self.PLAYER_ATTACK_DURATION = 8  # frames
        self.PLAYER_ATTACK_COOLDOWN = 15
        self.PLAYER_ATTACK_RANGE = 60
        self.PLAYER_INVINCIBILITY_DURATION = 30 # frames
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 0
        self.is_grounded = False
        self.attack_timer = 0
        self.attack_cooldown = 0
        self.invincibility_timer = 0
        self.prev_space_held = False

        self.monsters = []
        self.particles = []
        self.floating_texts = []
        
        self.camera_x = 0
        self.camera_shake = 0

        self.wave = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.np_random = None # Will be initialized in reset
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False

        self.player_pos = pygame.Vector2(self.W / 2, self.GROUND_Y)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 50
        self.is_grounded = True
        self.attack_timer = 0
        self.attack_cooldown = 0
        self.invincibility_timer = 0
        self.prev_space_held = False

        self.monsters.clear()
        self.particles.clear()
        self.floating_texts.clear()
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for each step to encourage speed

        # --- Update Game Logic ---
        self._handle_input(movement, space_held)
        reward += self._update_player(movement)
        self._update_monsters()
        
        # Collision detection and corresponding rewards
        reward += self._handle_collisions()

        self._update_effects()
        self._update_camera()

        self.steps += 1
        terminated = (self.player_health <= 0 or 
                      len(self.monsters) == 0 or 
                      self.steps >= self.MAX_STEPS)

        if self.player_health <= 0 and not self.game_over:
            reward -= 100
            self.game_over = True
        elif len(self.monsters) == 0 and not self.game_over:
            reward += 100
            # For this implementation, clearing a wave ends the episode.
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Horizontal Movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x *= 0.8  # Friction

        # Jumping
        if movement == 1 and self.is_grounded:  # Up
            self.player_vel.y = -self.JUMP_STRENGTH
            self.is_grounded = False
            # Sound: Player Jump
            self._create_particles(self.player_pos + pygame.Vector2(0, 10), 5, self.COLOR_GROUND, 0.5)

        # Attacking
        if space_held and not self.prev_space_held and self.attack_cooldown <= 0:
            self.attack_timer = self.PLAYER_ATTACK_DURATION
            self.attack_cooldown = self.PLAYER_ATTACK_COOLDOWN
            # Sound: Player Attack Swing
        self.prev_space_held = space_held

    def _update_player(self, movement):
        # Update timers
        if self.attack_timer > 0: self.attack_timer -= 1
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        if self.invincibility_timer > 0: self.invincibility_timer -= 1

        # Apply physics
        self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel

        # Ground collision
        if self.player_pos.y >= self.GROUND_Y:
            if not self.is_grounded: # Landing effect
                self._create_particles(self.player_pos + pygame.Vector2(0, 10), 8, self.COLOR_GROUND, 0.7)
            self.player_pos.y = self.GROUND_Y
            self.player_vel.y = 0
            self.is_grounded = True

        # World boundaries
        self.player_pos.x = max(10, min(self.player_pos.x, self.LEVEL_WIDTH - 10))

        # Proximity reward
        if self.monsters:
            nearest_monster = min(self.monsters, key=lambda m: abs(m['pos'].x - self.player_pos.x))
            is_moving_towards = (
                (movement == 4 and nearest_monster['pos'].x > self.player_pos.x) or
                (movement == 3 and nearest_monster['pos'].x < self.player_pos.x)
            )
            if is_moving_towards:
                return 0.1
        return 0

    def _update_monsters(self):
        for monster in self.monsters:
            # Simple AI: move towards player
            direction = (self.player_pos - monster['pos']).normalize() if (self.player_pos - monster['pos']).length() > 0 else pygame.Vector2(0,0)
            monster['vel'] = direction * monster['speed']
            monster['pos'] += monster['vel']

            # Basic gravity
            if monster['pos'].y < self.GROUND_Y:
                monster['pos'].y += self.GRAVITY * 2
            monster['pos'].y = min(monster['pos'].y, self.GROUND_Y)

    def _handle_collisions(self):
        reward = 0
        
        # Player attack vs Monsters
        if self.attack_timer > 0:
            attack_rect_center = self.player_pos + pygame.Vector2(self.PLAYER_ATTACK_RANGE / 2 * (1 if self.player_vel.x >= 0 else -1), 0)
            attack_rect = pygame.Rect(0, 0, self.PLAYER_ATTACK_RANGE, 30)
            attack_rect.center = attack_rect_center

            for monster in self.monsters[:]:
                if not monster.get('hit_this_swing', False):
                    monster_rect = pygame.Rect(0, 0, 20, 30)
                    monster_rect.midbottom = monster['pos']
                    if attack_rect.colliderect(monster_rect):
                        monster['health'] -= 1
                        monster['hit_this_swing'] = True
                        reward += 1
                        self.score += 10
                        self._create_particles(monster['pos'], 10, self.COLOR_MONSTER, 1.5)
                        self._create_floating_text("+10", monster['pos'], (255, 255, 100))
                        # Sound: Monster Hit
                        if monster['health'] <= 0:
                            self.monsters.remove(monster)
                            self.score += 50
                            self._create_floating_text("+50", monster['pos'], (255, 200, 50))
                            # Sound: Monster Defeated
        else: # Reset hit flag after swing is over
            for m in self.monsters: m['hit_this_swing'] = False

        # Monsters vs Player
        if self.invincibility_timer <= 0:
            player_rect = pygame.Rect(0, 0, 20, 40)
            player_rect.midbottom = self.player_pos
            for monster in self.monsters:
                monster_rect = pygame.Rect(0, 0, 20, 30)
                monster_rect.midbottom = monster['pos']
                if player_rect.colliderect(monster_rect):
                    self.player_health -= 10
                    reward -= 1
                    self.invincibility_timer = self.PLAYER_INVINCIBILITY_DURATION
                    self.camera_shake = 10
                    self.player_vel.x += -5 if monster['pos'].x < self.player_pos.x else 5
                    self.player_vel.y = -5
                    # Sound: Player Damage
                    break
        return reward
    
    def _spawn_wave(self):
        monster_speed = 1.5 + (self.wave - 1) * 0.05
        monster_health = 2 + (self.wave - 1) * 1
        
        for _ in range(self.MONSTERS_PER_WAVE):
            side = self.np_random.choice([-1, 1])
            spawn_x = self.player_pos.x + side * self.np_random.integers(self.W // 2, self.W)
            spawn_x = max(20, min(spawn_x, self.LEVEL_WIDTH - 20))
            
            self.monsters.append({
                'pos': pygame.Vector2(spawn_x, self.GROUND_Y - 100),
                'vel': pygame.Vector2(0, 0),
                'health': monster_health,
                'speed': monster_speed * self.np_random.uniform(0.8, 1.2),
            })
        assert len(self.monsters) == self.MONSTERS_PER_WAVE

    def _update_effects(self):
        # Particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Floating Texts
        for ft in self.floating_texts[:]:
            ft['pos'].y -= 1
            ft['life'] -= 1
            if ft['life'] <= 0:
                self.floating_texts.remove(ft)

        # Camera Shake
        if self.camera_shake > 0:
            self.camera_shake -= 1

    def _update_camera(self):
        target_camera_x = self.player_pos.x - self.W / 2
        self.camera_x = max(0, min(target_camera_x, self.LEVEL_WIDTH - self.W))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Camera shake offset
        shake_offset = pygame.Vector2(0, 0)
        if self.camera_shake > 0:
            shake_offset.x = self.np_random.integers(-5, 6)
            shake_offset.y = self.np_random.integers(-5, 6)
        
        cam_x = self.camera_x - shake_offset.x
        
        # Render parallax background
        self._render_parallax(cam_x)
        
        # Render ground
        ground_rect = pygame.Rect(0, self.GROUND_Y, self.W, self.H - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

        # Render game elements
        self._render_monsters(cam_x)
        self._render_player(cam_x)
        self._render_particles(cam_x)
        
        # Render UI
        self._render_ui()
        self._render_floating_texts(cam_x)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_parallax(self, cam_x):
        # Far layer
        for i in range(-1, int(self.LEVEL_WIDTH / 200) + 2):
            x = (i * 200 - cam_x * 0.2) % (self.W + 200) - 100
            pygame.draw.rect(self.screen, self.COLOR_PARALLAX_1, (x, self.GROUND_Y - 80, 100, 80))
        # Mid layer
        for i in range(-1, int(self.LEVEL_WIDTH / 300) + 2):
            x = (i * 300 - cam_x * 0.5) % (self.W + 300) - 150
            pygame.draw.rect(self.screen, self.COLOR_PARALLAX_2, (x, self.GROUND_Y - 50, 150, 50))

    def _render_player(self, cam_x):
        # Invincibility flash
        if self.invincibility_timer > 0 and self.steps % 4 < 2:
            return

        screen_pos = self.player_pos - pygame.Vector2(cam_x, 0)
        
        # Simple animated sprite (squash and stretch)
        h = 40
        w = 20
        if not self.is_grounded: # In air
            h, w = 45, 18
        elif abs(self.player_vel.x) > 0.1: # Running
             w = 22 + math.sin(self.steps * 0.5) * 2
        
        player_rect = pygame.Rect(0, 0, w, h)
        player_rect.midbottom = (int(screen_pos.x), int(screen_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

        # Attack visual
        if self.attack_timer > 0:
            progress = 1 - (self.attack_timer / self.PLAYER_ATTACK_DURATION)
            angle = math.pi * progress
            direction = 1 if self.player_vel.x >= 0 else -1
            
            x1 = int(screen_pos.x)
            y1 = int(screen_pos.y - 20)
            x2 = int(x1 + math.cos(angle * direction - math.pi/2 * direction) * self.PLAYER_ATTACK_RANGE * 0.8)
            y2 = int(y1 + math.sin(angle * direction - math.pi/2 * direction) * self.PLAYER_ATTACK_RANGE * 0.8)
            
            pygame.draw.aaline(self.screen, self.COLOR_PLAYER_ATTACK, (x1, y1), (x2, y2), 2)

    def _render_monsters(self, cam_x):
        for monster in self.monsters:
            screen_pos = monster['pos'] - pygame.Vector2(cam_x, 0)
            monster_rect = pygame.Rect(0, 0, 20, 30)
            monster_rect.midbottom = (int(screen_pos.x), int(screen_pos.y))
            
            # Use color to indicate health
            health_ratio = monster['health'] / (2 + (self.wave - 1) * 1)
            color = self.COLOR_MONSTER
            if health_ratio < 0.5:
                color = (255, 150, 160)
            
            pygame.draw.rect(self.screen, color, monster_rect, border_radius=4)
    
    def _render_particles(self, cam_x):
        for p in self.particles:
            screen_pos = p['pos'] - pygame.Vector2(cam_x, 0)
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), size, color)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / 50)
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 10, 200 * health_ratio, 20))

        # Score
        score_surf = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.W - score_surf.get_width() - 10, 10))

        # Wave
        wave_surf = self.font_large.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_surf, (self.W // 2 - wave_surf.get_width() // 2, 10))
        
        # Termination message
        if self.game_over:
            msg = "WAVE CLEARED!" if len(self.monsters) == 0 else "GAME OVER"
            color = (100, 255, 150) if len(self.monsters) == 0 else (255, 100, 100)
            end_surf = self.font_large.render(msg, True, color)
            self.screen.blit(end_surf, (self.W // 2 - end_surf.get_width() // 2, self.H // 2 - end_surf.get_height() // 2))

    def _render_floating_texts(self, cam_x):
        for ft in self.floating_texts:
            alpha = int(255 * (ft['life'] / ft['max_life']))
            if alpha > 0:
                text_surf = self.font_small.render(ft['text'], True, ft['color'])
                text_surf.set_alpha(alpha)
                screen_pos = ft['pos'] - pygame.Vector2(cam_x, 0)
                self.screen.blit(text_surf, (int(screen_pos.x), int(screen_pos.y)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
            "monsters_left": len(self.monsters),
        }
    
    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _create_floating_text(self, text, pos, color):
        life = 40
        self.floating_texts.append({
            'text': text,
            'pos': pos.copy(),
            'color': color,
            'life': life,
            'max_life': life
        })
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # Test specific game assertions
        self.reset()
        assert self.player_health <= 50
        assert len(self.monsters) == self.MONSTERS_PER_WAVE
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Monster Wave Survivor")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print(env.user_guide)

    while not terminated:
        # --- Manual Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()