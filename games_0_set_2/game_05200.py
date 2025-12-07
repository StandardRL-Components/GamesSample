
# Generated: 2025-08-28T04:16:32.361878
# Source Brief: brief_05200.md
# Brief Index: 5200

        
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
        "Controls: ←→ to move. Press space to fire."
    )

    game_description = (
        "A minimalist top-down space shooter. Eliminate waves of descending aliens while dodging their projectiles."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        self.MAX_WAVES = 5
        self.STARTING_LIVES = 3
        
        # Colors
        self.COLOR_BG = (10, 10, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PLAYER_BULLET = (200, 255, 255)
        self.COLOR_ENEMY_BULLET = (255, 100, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        self.COLOR_PARTICLE_EXPLOSION = [(255, 50, 50), (255, 150, 50), (200, 200, 200)]
        self.COLOR_PARTICLE_HIT = [(200, 200, 255), (255, 255, 255)]

        # Player
        self.PLAYER_SPEED = 6
        self.PLAYER_FIRE_COOLDOWN = 6 # 5 shots per second at 30fps

        # Bullets
        self.BULLET_SPEED = 10
        
        # Aliens
        self.ALIEN_BASE_SPEED = 0.3
        self.ALIEN_SPEED_WAVE_INC = 0.2
        self.ALIEN_BASE_FIRE_RATE = 0.02 # prob per frame
        self.ALIEN_FIRE_RATE_WAVE_INC = 0.005

        # Rewards
        self.REWARD_DESTROY_ALIEN = 10.0
        self.REWARD_RISKY_KILL = 2.0
        self.REWARD_LOSE_LIFE = -10.0
        self.REWARD_STATIONARY = -2.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -100.0
        self.REWARD_PER_STEP = -0.01

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.Font(None, 28)
        self.title_font = pygame.font.Font(None, 64)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.np_random = None

        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_fire_cooldown_timer = 0
        self.stationary_frames = 0
        
        self.current_wave = 0
        self.wave_transition_timer = 0

        self.aliens = []
        self.player_bullets = []
        self.enemy_bullets = []
        self.particles = []
        
        self.reset()
        
        # --- Validation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_lives = self.STARTING_LIVES
        self.player_fire_cooldown_timer = 0
        self.stationary_frames = 0
        
        self.aliens.clear()
        self.player_bullets.clear()
        self.enemy_bullets.clear()
        self.particles.clear()
        
        self.current_wave = 1
        self.wave_transition_timer = 90 # 3 seconds at 30fps
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = self.REWARD_PER_STEP
        terminated = False

        if self.game_over or self.game_won:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- Handle Wave Transitions ---
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer == 0:
                self._spawn_wave()
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Unpack Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self._handle_player_input(movement, space_held)
        self._update_bullets()
        self._update_aliens()
        self._update_particles()
        
        # --- Collision Detection and Rewards ---
        reward += self._handle_collisions()
        
        # --- Stationary Penalty ---
        if movement == 0:
            self.stationary_frames += 1
            if self.stationary_frames > 10:
                reward += self.REWARD_STATIONARY
        else:
            self.stationary_frames = 0
            
        # --- Check Termination Conditions ---
        if self.player_lives <= 0:
            self.game_over = True
            terminated = True
            reward += self.REWARD_LOSE
            # SFX: Game over
            
        elif not self.aliens and self.wave_transition_timer == 0:
            if self.current_wave == self.MAX_WAVES:
                self.game_won = True
                terminated = True
                reward += self.REWARD_WIN
                # SFX: Game win
            else:
                self.current_wave += 1
                self.wave_transition_timer = 90 # Start next wave transition

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Movement
        if movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        
        # Firing
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        
        if space_held and self.player_fire_cooldown_timer == 0:
            bullet_pos = [self.player_pos[0], self.player_pos[1] - 15]
            self.player_bullets.append(pygame.Rect(bullet_pos[0]-1, bullet_pos[1], 3, 10))
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN
            # SFX: Player shoot

    def _update_bullets(self):
        self.player_bullets = [b for b in self.player_bullets if b.y > -10]
        for bullet in self.player_bullets:
            bullet.y -= self.BULLET_SPEED
            
        self.enemy_bullets = [b for b in self.enemy_bullets if b.y < self.HEIGHT + 10]
        for bullet in self.enemy_bullets:
            bullet.y += self.BULLET_SPEED

    def _update_aliens(self):
        alien_speed = self.ALIEN_BASE_SPEED + (self.current_wave - 1) * self.ALIEN_SPEED_WAVE_INC
        fire_prob = self.ALIEN_BASE_FIRE_RATE + (self.current_wave - 1) * self.ALIEN_FIRE_RATE_WAVE_INC

        for alien in self.aliens:
            alien['pos'][1] += alien_speed
            alien['rect'].topleft = alien['pos']

            if self.np_random.random() < fire_prob:
                bullet_pos = [alien['rect'].centerx, alien['rect'].bottom]
                self.enemy_bullets.append(pygame.Rect(bullet_pos[0]-1, bullet_pos[1], 3, 10))
                # SFX: Enemy shoot
    
    def _handle_collisions(self):
        reward = 0
        
        # Player bullets vs Aliens
        aliens_to_remove = []
        bullets_to_remove = []
        for i, bullet in enumerate(self.player_bullets):
            for j, alien in enumerate(self.aliens):
                if j in aliens_to_remove: continue
                if bullet.colliderect(alien['rect']):
                    aliens_to_remove.append(j)
                    if i not in bullets_to_remove: bullets_to_remove.append(i)
                    
                    self.score += int(self.REWARD_DESTROY_ALIEN)
                    reward += self.REWARD_DESTROY_ALIEN
                    
                    # Risky play bonus
                    dist = math.hypot(self.player_pos[0] - alien['rect'].centerx, self.player_pos[1] - alien['rect'].centery)
                    if dist < 100:
                        reward += self.REWARD_RISKY_KILL
                        
                    self._create_particles(alien['rect'].center, 20, self.COLOR_PARTICLE_EXPLOSION)
                    # SFX: Alien explosion
                    break
        
        self.aliens = [a for i, a in enumerate(self.aliens) if i not in aliens_to_remove]
        self.player_bullets = [b for i, b in enumerate(self.player_bullets) if i not in bullets_to_remove]

        # Enemy bullets vs Player
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 8, 20, 16)
        for bullet in self.enemy_bullets:
            if player_rect.colliderect(bullet):
                self.enemy_bullets.remove(bullet)
                self.player_lives -= 1
                reward += self.REWARD_LOSE_LIFE
                self._create_particles(self.player_pos, 30, self.COLOR_PARTICLE_HIT)
                # SFX: Player hit
                break # only one hit per frame
                
        return reward
    
    def _spawn_wave(self):
        self.aliens.clear()
        rows = 2 + (self.current_wave // 2)
        cols = 6 + self.current_wave
        
        x_spacing = (self.WIDTH - 100) / (cols -1) if cols > 1 else 0
        y_spacing = 40
        
        for r in range(rows):
            for c in range(cols):
                x = 50 + c * x_spacing
                y = -50 - r * y_spacing
                alien_rect = pygame.Rect(x-10, y-10, 20, 20)
                self.aliens.append({'rect': alien_rect, 'pos': [x-10, y-10]})

    def _create_particles(self, pos, count, colors):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 3
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = 15 + self.np_random.integers(0, 15)
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life,
                'color': random.choice(colors)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color']
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], alpha))
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))
            
        # Bullets
        for bullet in self.enemy_bullets:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_BULLET, bullet)
        for bullet in self.player_bullets:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_BULLET, bullet)
            
        # Aliens
        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, alien['rect'])
            
        # Player
        if self.player_lives > 0:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            points = [(px, py - 10), (px - 10, py + 8), (px + 10, py + 8)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(2,2)):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score and Wave
        draw_text(f"SCORE: {self.score}", self.ui_font, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        wave_str = f"WAVE: {self.current_wave}/{self.MAX_WAVES}"
        text_w = self.ui_font.size(wave_str)[0]
        draw_text(wave_str, self.ui_font, self.COLOR_TEXT, (self.WIDTH - text_w - 10, 10), self.COLOR_TEXT_SHADOW)

        # Lives
        for i in range(self.player_lives):
            px, py = 30 + i * 25, self.HEIGHT - 20
            points = [(px, py - 6), (px - 6, py + 5), (px + 6, py + 5)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Game Status Text
        if self.game_over:
            text = "GAME OVER"
            text_w, text_h = self.title_font.size(text)
            draw_text(text, self.title_font, self.COLOR_ENEMY, (self.WIDTH/2 - text_w/2, self.HEIGHT/2 - text_h/2), self.COLOR_TEXT_SHADOW)
        elif self.game_won:
            text = "YOU WIN!"
            text_w, text_h = self.title_font.size(text)
            draw_text(text, self.title_font, (100, 255, 100), (self.WIDTH/2 - text_w/2, self.HEIGHT/2 - text_h/2), self.COLOR_TEXT_SHADOW)
        elif self.wave_transition_timer > 0:
            text = f"WAVE {self.current_wave}"
            text_w, text_h = self.title_font.size(text)
            draw_text(text, self.title_font, self.COLOR_TEXT, (self.WIDTH/2 - text_w/2, self.HEIGHT/2 - text_h/2), self.COLOR_TEXT_SHADOW)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Note: Requires pygame to be installed with display support.
    #       `pip install pygame`
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    try:
        display_screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Space Shooter")
    except pygame.error:
        print("Pygame display could not be initialized. Running headlessly.")
        display_screen = None

    running = True
    total_reward = 0
    
    # Set a default action
    action = env.action_space.sample()
    action[0] = 0 # No movement
    action[1] = 0 # Space not held
    action[2] = 0 # Shift not held

    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("Press ESC or close the window to quit.")
    print("="*30 + "\n")

    while running:
        # --- Human Input ---
        if display_screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            
            # Reset action
            action[0] = 0 # No movement
            action[1] = 0 # Space not held
            
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_SPACE]:
                action[1] = 1

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if display_screen:
            # Transpose back for pygame display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            if display_screen:
                pygame.time.wait(2000) # Pause before restarting

    env.close()