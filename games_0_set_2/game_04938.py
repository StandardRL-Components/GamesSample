# Generated: 2025-08-28T03:29:26.520945
# Source Brief: brief_04938.md
# Brief Index: 4938

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    """
    A retro arcade shooter environment where the player defends against a descending alien horde.
    The player controls a ship at the bottom of the screen, moving horizontally and firing
    projectiles upwards to destroy aliens. The game progresses through stages and waves of
    increasing difficulty.
    """
    # Metadata and descriptions
    metadata = {"render_modes": ["rgb_array"]}
    user_guide = "Controls: ←→ to move. Press space to fire."
    game_description = "Defend Earth from a descending alien horde in this retro arcade shooter."
    auto_advance = True

    # --- Constants ---
    # Screen and Layout
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 40
    PLAY_AREA_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT
    BORDER_WIDTH = 5

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_GRID = (20, 15, 40)
    COLOR_BORDER = (50, 50, 80)
    COLOR_TEXT = (220, 220, 255)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_PROJECTILE = (255, 255, 255)
    COLOR_ALIEN_STD = (255, 50, 50)
    COLOR_ALIEN_BOMBER = (255, 150, 50)
    COLOR_ALIEN_PROJECTILE = (255, 100, 255)
    EXPLOSION_COLORS = [(255, 255, 0), (255, 150, 0), (255, 50, 0)]

    # Player
    PLAYER_WIDTH = 28
    PLAYER_HEIGHT = 14
    PLAYER_SPEED = 8
    PLAYER_SHOOT_COOLDOWN = 8  # frames
    PLAYER_INVINCIBILITY_DURATION = 90 # frames (3 seconds)

    # Projectiles
    PROJECTILE_SPEED = 12
    PROJECTILE_WIDTH = 4
    PROJECTILE_HEIGHT = 12

    # Aliens
    ALIEN_COLS = 10
    ALIEN_SIZE = 20
    ALIEN_H_SPACING = 35
    ALIEN_V_SPACING = 30
    INITIAL_ALIEN_V_SPEED = 0.1
    ALIEN_SPEED_WAVE_INCREMENT = 0.05
    ALIEN_DROP_AMOUNT = 10
    ALIEN_BOMBER_CHANCE = 0.002 # Per alien per frame

    # Game Flow
    MAX_LIVES = 3
    STAGES = 3
    WAVES_PER_STAGE = [3, 4, 5]
    STEPS_PER_STAGE = 1800 # 60s * 30fps
    MAX_EPISODE_STEPS = STEPS_PER_STAGE * STAGES + 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Set the SDL video driver to dummy for headless operation
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        self.game_over_font = pygame.font.Font(None, 72)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - self.PLAYER_HEIGHT - 10)
        self.player_lives = self.MAX_LIVES
        self.player_shoot_cooldown = 0
        self.player_invincibility_timer = 0

        self.player_projectiles = []
        self.alien_projectiles = []
        self.aliens = []
        self.particles = []

        self.current_stage = 1
        self.current_wave = 1
        self.stage_timer = self.STEPS_PER_STAGE

        self.alien_h_speed = self.INITIAL_ALIEN_V_SPEED * 10
        self.alien_v_speed = self.INITIAL_ALIEN_V_SPEED
        self.alien_move_direction = 1

        self.space_was_held = False

        self._setup_wave()

        return self._get_observation(), self._get_info()

    def _setup_wave(self):
        self.aliens.clear()
        self.alien_projectiles.clear()
        self.player_projectiles.clear()
        
        difficulty_mod = (self.current_stage - 1) * 0.5 + (self.current_wave - 1) * 0.2
        self.alien_v_speed = self.INITIAL_ALIEN_V_SPEED + difficulty_mod * self.ALIEN_SPEED_WAVE_INCREMENT
        self.alien_h_speed = self.alien_v_speed * 10

        rows = self.WAVES_PER_STAGE[self.current_stage - 1]
        start_x = (self.SCREEN_WIDTH - (self.ALIEN_COLS * self.ALIEN_H_SPACING)) / 2
        start_y = self.UI_HEIGHT + 40

        for r in range(rows):
            for c in range(self.ALIEN_COLS):
                x = start_x + c * self.ALIEN_H_SPACING
                y = start_y + r * self.ALIEN_V_SPACING
                
                alien_type = "standard"
                if self.current_stage > 1 and r < 2:
                    if self.np_random.random() < 0.4:
                        alien_type = "bomber"

                self.aliens.append({
                    "pos": pygame.Vector2(x, y),
                    "type": alien_type,
                    "size": self.ALIEN_SIZE,
                })

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = 0
        terminated = False
        truncated = False
        
        if not self.game_over:
            self._handle_input(action)
            self._update_aliens()
            reward += self._update_projectiles()
            self._update_particles()
            
            flow_reward, flow_terminated = self._update_game_flow()
            reward += flow_reward
            if flow_terminated:
                terminated = True
        
        if self.player_lives <= 0:
            if not self.game_over:
                 self.game_over = True
            terminated = True
            
        self.steps += 1
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            terminated = True # Per Gymnasium API, truncated episodes are also terminated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        self.player_pos.x = np.clip(
            self.player_pos.x, 
            self.PLAYER_WIDTH / 2 + self.BORDER_WIDTH, 
            self.SCREEN_WIDTH - self.PLAYER_WIDTH / 2 - self.BORDER_WIDTH
        )
        
        if self.player_shoot_cooldown > 0:
            self.player_shoot_cooldown -= 1
        
        if space_held and not self.space_was_held and self.player_shoot_cooldown <= 0:
            self.player_projectiles.append(pygame.Rect(
                self.player_pos.x - self.PROJECTILE_WIDTH / 2,
                self.player_pos.y - self.PLAYER_HEIGHT,
                self.PROJECTILE_WIDTH,
                self.PROJECTILE_HEIGHT
            ))
            self.player_shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN
            # sfx: player_shoot.wav
            
        self.space_was_held = space_held
        
        if self.player_invincibility_timer > 0:
            self.player_invincibility_timer -= 1

    def _update_aliens(self):
        if not self.aliens:
            return

        move_down = False
        for alien in self.aliens:
            alien["pos"].y += self.alien_v_speed
            alien["pos"].x += self.alien_h_speed * self.alien_move_direction
            
            if (alien["pos"].x <= self.BORDER_WIDTH + self.ALIEN_SIZE / 2 or 
                alien["pos"].x >= self.SCREEN_WIDTH - self.BORDER_WIDTH - self.ALIEN_SIZE / 2):
                move_down = True
            
            if alien["pos"].y > self.player_pos.y - self.ALIEN_SIZE:
                self.game_over = True
                self.player_lives = 0
            
            if alien["type"] == "bomber" and self.np_random.random() < self.ALIEN_BOMBER_CHANCE:
                self.alien_projectiles.append(pygame.Rect(
                    alien["pos"].x - self.PROJECTILE_WIDTH / 2,
                    alien["pos"].y,
                    self.PROJECTILE_WIDTH,
                    self.PROJECTILE_HEIGHT
                ))
                # sfx: alien_shoot.wav
                
        if move_down:
            self.alien_move_direction *= -1
            for alien in self.aliens:
                alien["pos"].y += self.ALIEN_DROP_AMOUNT
                alien["pos"].x += self.alien_h_speed * self.alien_move_direction

    def _update_projectiles(self):
        reward = 0
        player_hitbox = pygame.Rect(
            self.player_pos.x - self.PLAYER_WIDTH / 2,
            self.player_pos.y - self.PLAYER_HEIGHT / 2,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT
        )

        for proj in self.player_projectiles[:]:
            proj.y -= self.PROJECTILE_SPEED
            if proj.bottom < self.UI_HEIGHT:
                self.player_projectiles.remove(proj)
                reward -= 0.02
                continue

            hit_alien = False
            for alien in self.aliens[:]:
                alien_hitbox = pygame.Rect(
                    alien["pos"].x - alien["size"] / 2,
                    alien["pos"].y - alien["size"] / 2,
                    alien["size"], alien["size"]
                )
                if proj.colliderect(alien_hitbox):
                    self.aliens.remove(alien)
                    self.player_projectiles.remove(proj)
                    reward += 1.0
                    self.score += 10
                    self._create_explosion(alien["pos"])
                    # sfx: explosion.wav
                    hit_alien = True
                    break
            if hit_alien:
                continue

        for proj in self.alien_projectiles[:]:
            proj.y += self.PROJECTILE_SPEED / 2
            if proj.top > self.SCREEN_HEIGHT:
                self.alien_projectiles.remove(proj)
                continue
            
            if self.player_invincibility_timer <= 0 and proj.colliderect(player_hitbox):
                self.alien_projectiles.remove(proj)
                self.player_lives -= 1
                reward -= 10.0
                self.player_invincibility_timer = self.PLAYER_INVINCIBILITY_DURATION
                self._create_explosion(self.player_pos, is_player=True)
                # sfx: player_hit.wav
                if self.player_lives > 0:
                    self.player_pos.x = self.SCREEN_WIDTH / 2
                continue
        return reward

    def _update_game_flow(self):
        reward = 0
        terminated = False
        
        self.stage_timer -= 1
        if self.stage_timer <= 0 and self.current_stage <= self.STAGES:
            reward += 100
            self.current_stage += 1
            if self.current_stage > self.STAGES:
                self.game_won = True
                terminated = True
                reward += 500
            else:
                self.current_wave = 1
                self.stage_timer = self.STEPS_PER_STAGE
                self._setup_wave()
            return reward, terminated

        if not self.aliens and not self.game_won:
            self.current_wave += 1
            if self.current_wave > self.WAVES_PER_STAGE[self.current_stage - 1]:
                reward += 100
                self.current_stage += 1
                if self.current_stage > self.STAGES:
                    self.game_won = True
                    terminated = True
                    reward += 500
                else:
                    self.current_wave = 1
                    self.stage_timer = self.STEPS_PER_STAGE
                    self._setup_wave()
            else:
                self._setup_wave()
                
        return reward, terminated

    def _create_explosion(self, pos, count=20, is_player=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5 if not is_player else 8)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": random.choice(self.EXPLOSION_COLORS),
                "size": self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_game()
        self._render_ui()

        if self.game_over and self.player_lives <= 0:
            text_surf = self.game_over_font.render("GAME OVER", True, self.COLOR_ALIEN_STD)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
        elif self.game_won:
            text_surf = self.game_over_font.render("YOU WIN!", True, self.COLOR_PLAYER)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.UI_HEIGHT), (x, self.SCREEN_HEIGHT))
        for y in range(self.UI_HEIGHT, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
    
    def _render_game(self):
        if self.player_lives > 0:
            self._render_player()

        for alien in self.aliens:
            self._render_alien(alien)

        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, proj)
        for proj in self.alien_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJECTILE, proj)

        for p in self.particles:
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y), p["size"], p["size"]))

    def _render_player(self):
        is_invincible = self.player_invincibility_timer > 0
        if is_invincible and (self.player_invincibility_timer % 10 < 5):
            return # Blink effect
        
        points = [
            (self.player_pos.x, self.player_pos.y - self.PLAYER_HEIGHT / 2),
            (self.player_pos.x + self.PLAYER_WIDTH / 2, self.player_pos.y + self.PLAYER_HEIGHT / 2),
            (self.player_pos.x - self.PLAYER_WIDTH / 2, self.player_pos.y + self.PLAYER_HEIGHT / 2)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

    def _render_alien(self, alien):
        color = self.COLOR_ALIEN_STD if alien["type"] == "standard" else self.COLOR_ALIEN_BOMBER
        size = int(alien["size"])
        pos = alien["pos"]
        
        body = pygame.Rect(int(pos.x - size/2), int(pos.y - size/2), size, size)
        pygame.draw.rect(self.screen, color, body)
        pygame.draw.rect(self.screen, self.COLOR_BG, (int(pos.x - size/4), int(pos.y), int(size/2), int(size/4)))
        pygame.draw.rect(self.screen, (255,255,255), (int(pos.x - size/3), int(pos.y - size/4), 2, 2))
        pygame.draw.rect(self.screen, (255,255,255), (int(pos.x + size/3 - 2), int(pos.y - size/4), 2, 2))

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        score_text = self.font_large.render(f"SCORE: {self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 8))
        
        lives_text = self.font_large.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 200, 8))
        for i in range(self.player_lives):
            points = [
                (self.SCREEN_WIDTH - 110 + i * 25, 12),
                (self.SCREEN_WIDTH - 100 + i * 25, 28),
                (self.SCREEN_WIDTH - 120 + i * 25, 28)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
            
        stage_text = self.font_small.render(f"STAGE: {self.current_stage}-{self.current_wave}", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(centerx=self.SCREEN_WIDTH/2, centery=self.UI_HEIGHT/2)
        self.screen.blit(stage_text, stage_rect)

        play_area = pygame.Rect(0, self.UI_HEIGHT, self.SCREEN_WIDTH, self.PLAY_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BORDER, play_area, self.BORDER_WIDTH)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "stage": self.current_stage,
            "wave": self.current_wave,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Set the SDL video driver to a visible one for human play
    os.environ['SDL_VIDEODRIVER'] = 'x11'
    
    env = GameEnv(render_mode="rgb_array")
    # The validate_implementation method is for developer use, not part of the standard Env API
    # It's called here to ensure the environment is correctly set up before running.
    # env.validate_implementation() # Temporarily disabled for standard run

    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Grid Defender")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
    env.close()