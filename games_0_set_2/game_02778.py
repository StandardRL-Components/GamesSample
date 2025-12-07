
# Generated: 2025-08-27T21:24:03.148553
# Source Brief: brief_02778.md
# Brief Index: 2778

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric tower defense game where the player must survive 10 waves of enemies.
    The player controls a central tower and can unleash a single area-of-effect attack
    to destroy incoming enemies. Timing is crucial to maximize damage and conserve health.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = "Press space to fire your weapon."

    # User-facing description of the game
    game_description = "Survive 10 waves of enemies by strategically timing your single attack in this isometric tower defense game."

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 3000  # 100 seconds at 30fps

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 50)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ATTACK = (255, 255, 0)
    COLOR_PARTICLE = (255, 100, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (0, 200, 100)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_COOLDOWN = (0, 150, 255)

    # --- Player/Tower Constants ---
    PLAYER_POS = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2], dtype=float)
    PLAYER_RADIUS = 12
    PLAYER_MAX_HEALTH = 100

    # --- Attack Constants ---
    ATTACK_RADIUS = 120
    ATTACK_DURATION = 10  # frames
    ATTACK_COOLDOWN = 45 # frames (1.5 seconds)

    # --- Wave Constants ---
    MAX_WAVES = 10
    WAVE_START_DELAY = 90  # frames (3 seconds)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        self.render_mode = render_mode
        self.np_random = None # Will be seeded in reset

        # Initialize state variables (these are reset in self.reset())
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_health = 0
        self.current_wave = 0
        self.enemies = []
        self.particles = []
        self.attack_cooldown_timer = 0
        self.attack_animation_timer = 0
        self.wave_timer = 0

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_health = self.PLAYER_MAX_HEALTH
        self.current_wave = 0
        self.enemies = []
        self.particles = []
        self.attack_cooldown_timer = 0
        self.attack_animation_timer = 0
        self.wave_timer = self.WAVE_START_DELAY

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_pressed = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Wave Logic ---
        if not self.enemies and not self.win:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()
                if self.win:
                    terminated = True
                    reward += 100
                    self.game_over = True

        # --- Handle Player Action ---
        if space_pressed and self.attack_cooldown_timer <= 0:
            # SFX: Player Attack Whoosh
            self.attack_cooldown_timer = self.ATTACK_COOLDOWN
            self.attack_animation_timer = self.ATTACK_DURATION
            reward += self._process_attack()

        # --- Update Game State ---
        self._update_timers()
        self._update_enemies()
        self._update_particles()
        
        # --- Check for Damage to Player ---
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            dist_to_player = np.linalg.norm(enemy['pos'] - self.PLAYER_POS)
            if dist_to_player < self.PLAYER_RADIUS:
                # SFX: Player Damage Taken
                self.player_health -= enemy['damage']
                reward -= 1
                enemies_to_remove.append(i)
        
        if enemies_to_remove:
            self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]
            self.player_health = max(0, self.player_health)

        # --- Check Termination Conditions ---
        self.steps += 1
        if self.player_health <= 0:
            # SFX: Game Over Failure
            terminated = True
            reward -= 100
            self.game_over = True
        elif self.steps >= self.MAX_STEPS and not self.win:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            # SFX: Game Win Fanfare
            self.win = True
            return

        num_enemies = 2 + self.current_wave
        enemy_speed = 0.8 + self.current_wave * 0.1
        enemy_health = 1 + self.current_wave // 3
        
        for _ in range(num_enemies):
            side = self.np_random.integers(0, 4)
            if side == 0:  # Top
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -20.0])
            elif side == 1:  # Right
                pos = np.array([self.SCREEN_WIDTH + 20.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
            elif side == 2:  # Bottom
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20.0])
            else:  # Left
                pos = np.array([-20.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)])

            self.enemies.append({
                'pos': pos,
                'health': enemy_health,
                'speed': enemy_speed,
                'damage': 10 # Deals 10 damage
            })
        
        self.wave_timer = self.WAVE_START_DELAY

    def _process_attack(self):
        reward = 0
        enemies_hit_this_frame = []
        for i, enemy in enumerate(self.enemies):
            dist_to_player = np.linalg.norm(enemy['pos'] - self.PLAYER_POS)
            if dist_to_player < self.ATTACK_RADIUS:
                # SFX: Enemy Hit
                enemy['health'] -= 1
                reward += 0.1
                enemies_hit_this_frame.append(i)
        
        enemies_to_remove = []
        for i in enemies_hit_this_frame:
            if self.enemies[i]['health'] <= 0:
                # SFX: Enemy Death Explosion
                self._create_explosion(self.enemies[i]['pos'])
                reward += 1
                self.score += 10
                enemies_to_remove.append(i)

        if enemies_to_remove:
            self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]
        
        return reward

    def _update_timers(self):
        if self.attack_cooldown_timer > 0:
            self.attack_cooldown_timer -= 1
        if self.attack_animation_timer > 0:
            self.attack_animation_timer -= 1

    def _update_enemies(self):
        for enemy in self.enemies:
            direction_vec = self.PLAYER_POS - enemy['pos']
            dist = np.linalg.norm(direction_vec)
            if dist > 1:
                direction_norm = direction_vec / dist
                enemy['pos'] += direction_norm * enemy['speed']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'] *= 0.95 # Drag

    def _create_explosion(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw attack animation
        if self.attack_animation_timer > 0:
            progress = 1 - (self.attack_animation_timer / self.ATTACK_DURATION)
            current_radius = int(self.ATTACK_RADIUS * progress)
            alpha = int(255 * (1 - progress**2))
            
            # Use gfxdraw for anti-aliased shapes
            pygame.gfxdraw.aacircle(self.screen, int(self.PLAYER_POS[0]), int(self.PLAYER_POS[1]), current_radius, self.COLOR_ATTACK + (alpha,))
            pygame.gfxdraw.filled_circle(self.screen, int(self.PLAYER_POS[0]), int(self.PLAYER_POS[1]), current_radius, self.COLOR_ATTACK + (alpha // 3,))


        # Draw player tower
        px, py = int(self.PLAYER_POS[0]), int(self.PLAYER_POS[1])
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS + 5, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, ex, ey, 7, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, 7, self.COLOR_ENEMY)

        # Draw particles
        for p in self.particles:
            px, py = int(p['pos'][0]), int(p['pos'][1])
            alpha = int(255 * (p['life'] / 30))
            color = self.COLOR_PARTICLE + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(p['radius']), color)

    def _render_ui(self):
        # Health bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 100
        bar_height = 10
        bar_x = self.PLAYER_POS[0] - bar_width / 2
        bar_y = self.PLAYER_POS[1] - self.PLAYER_RADIUS - 20
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_ratio, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Cooldown indicator
        if self.attack_cooldown_timer > 0:
            cooldown_ratio = self.attack_cooldown_timer / self.ATTACK_COOLDOWN
            angle = 360 * cooldown_ratio
            rect = pygame.Rect(self.PLAYER_POS[0] - 20, self.PLAYER_POS[1] - 20, 40, 40)
            pygame.draw.arc(self.screen, self.COLOR_COOLDOWN, rect, math.pi / 2, math.pi / 2 + math.radians(angle), 2)

        # Wave counter
        wave_text = f"WAVE: {self.current_wave if not self.win else self.MAX_WAVES} / {self.MAX_WAVES}"
        text_surf = self.font_small.render(wave_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Game Over / Win Text
        if self.game_over:
            if self.win:
                message = "YOU WIN!"
                color = self.COLOR_PLAYER
            else:
                message = "GAME OVER"
                color = self.COLOR_ENEMY
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "current_wave": self.current_wave,
            "enemies_remaining": len(self.enemies)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and visualize the game
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        space_pressed = keys[pygame.K_SPACE]
        
        action = [0, 0, 0] # noop, released, released
        if space_pressed:
            action[1] = 1 # space held
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Survived to Wave: {info['current_wave']}")
            # obs, info = env.reset() # Uncomment to auto-reset

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()