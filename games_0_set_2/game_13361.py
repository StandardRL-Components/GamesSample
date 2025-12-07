import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:59:57.937057
# Source Brief: brief_03361.md
# Brief Index: 3361
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Assyrian Archery Alphabet: a Gymnasium environment where the player shoots
    lettered arrows at word targets to defeat advancing enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Shoot lettered arrows at word targets to defeat advancing enemies and defend your base."
    )
    user_guide = (
        "Controls: Use ↑ and ↓ arrow keys to aim your bow. Press space to shoot and shift to cycle letters."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    TOTAL_TERRITORIES = 10

    # Colors
    COLOR_BG = (18, 22, 33)
    COLOR_BASE = (60, 80, 120)
    COLOR_BASE_STROKE = (100, 120, 160)
    COLOR_PLAYER = (120, 200, 255)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_WORD = (255, 255, 255)
    COLOR_ENEMY_WORD_HIT = (100, 100, 100)
    COLOR_HEALTH_GREEN = (80, 220, 100)
    COLOR_HEALTH_RED = (220, 80, 80)
    COLOR_ARROW = (255, 220, 150)
    COLOR_ARROW_LETTER = (0, 0, 0)
    COLOR_PARTICLE_GOLD = (255, 215, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (255, 180, 90)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Game dictionaries and lists
        self.word_list = ["KING", "ARMY", "WAR", "BOW", "ARROW", "SHIELD", "FORT", "SIEGE", "EMPIRE", "CLAY", "TABLET"]
        self.base_letters = ['A', 'E', 'I', 'O', 'U', 'R', 'S', 'T', 'L', 'N']

        # Initialize state variables to be defined in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_base_health = None
        self.max_base_health = None
        self.aim_angle = None
        self.shoot_cooldown = None
        self.player_letters = None
        self.selected_letter_index = None
        self.combo_counter = None
        self.territories_conquered = None
        self.enemies = None
        self.arrows = None
        self.particles = None
        self.prev_space_held = None
        self.prev_shift_held = None
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for dev, not needed in prod

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.max_base_health = 100
        self.player_base_health = self.max_base_health
        self.aim_angle = 0  # Angle in degrees, 0 is horizontal right
        self.shoot_cooldown = 0
        
        self.player_letters = self.base_letters[:]
        self.selected_letter_index = 0
        
        self.combo_counter = 0
        self.territories_conquered = 0
        
        self.enemies = []
        self.arrows = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for time to encourage efficiency

        reward += self._handle_input(action)
        self._update_game_state()
        reward += self._check_collisions_and_process_hits()
        
        wave_cleared = len(self.enemies) == 0
        if wave_cleared:
            self.territories_conquered += 1
            reward += 100  # Big reward for conquering a territory
            # SFX: Territory Conquered Fanfare
            if self.territories_conquered < self.TOTAL_TERRITORIES:
                self._spawn_wave()
            else:
                self.game_over = True # Victory

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not truncated and self.player_base_health <= 0:
            reward = -100 # Penalty for losing
        elif terminated and not truncated and self.territories_conquered >= self.TOTAL_TERRITORIES:
            reward = 500 # Big reward for winning

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Aiming
        aim_speed = 3.0
        if movement == 1:  # Up
            self.aim_angle -= aim_speed
        elif movement == 2:  # Down
            self.aim_angle += aim_speed
        self.aim_angle = np.clip(self.aim_angle, -60, 60)

        # Cycle letter (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_letter_index = (self.selected_letter_index + 1) % len(self.player_letters)
            # SFX: UI Click
        
        # Shoot (on press)
        if space_held and not self.prev_space_held and self.shoot_cooldown <= 0:
            self._fire_arrow()
            self.shoot_cooldown = 10 # 1/3 of a second at 30fps
            self.combo_counter = 0 # Reset combo on new shot
            # SFX: Bow Twang
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return reward

    def _update_game_state(self):
        # Cooldowns
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        # Update arrows
        for arrow in self.arrows[:]:
            arrow['pos'][0] += arrow['vel'][0]
            arrow['pos'][1] += arrow['vel'][1]
            if arrow['pos'][0] > self.SCREEN_WIDTH:
                self.arrows.remove(arrow)

        # Update enemies
        for enemy in self.enemies[:]:
            enemy['pos'][0] -= enemy['speed']
            if enemy['pos'][0] < 100: # Reached the base
                self.player_base_health -= enemy['damage']
                self.enemies.remove(enemy)
                self._create_particles(pygame.Vector2(100, enemy['pos'][1]), self.COLOR_HEALTH_RED, 30)
                # SFX: Base Damage
                self.combo_counter = 0

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= 0.1
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _fire_arrow(self):
        angle_rad = math.radians(self.aim_angle)
        speed = 20.0
        start_pos = pygame.Vector2(100, self.SCREEN_HEIGHT / 2)
        velocity = pygame.Vector2(speed * math.cos(angle_rad), speed * math.sin(angle_rad))
        letter = self.player_letters[self.selected_letter_index]
        
        self.arrows.append({
            'pos': start_pos,
            'vel': velocity,
            'angle': self.aim_angle,
            'letter': letter
        })

    def _spawn_wave(self):
        num_enemies = min(3 + self.territories_conquered, 8)
        base_speed = 0.5 + self.territories_conquered * 0.1
        base_health = 50 + self.territories_conquered * 20

        for i in range(num_enemies):
            word = self.np_random.choice(self.word_list)
            self.enemies.append({
                'pos': pygame.Vector2(self.SCREEN_WIDTH + i * 80, self.np_random.integers(50, self.SCREEN_HEIGHT - 50)),
                'speed': base_speed + self.np_random.uniform(-0.1, 0.1),
                'max_health': base_health,
                'health': base_health,
                'word': list(word),
                'hit_letters': [False] * len(word),
                'damage': 20,
                'size': 20
            })

    def _check_collisions_and_process_hits(self):
        reward = 0
        for arrow in self.arrows[:]:
            arrow_rect = pygame.Rect(arrow['pos'].x, arrow['pos'].y, 10, 2)
            for enemy in self.enemies[:]:
                enemy_rect = pygame.Rect(enemy['pos'].x - enemy['size']/2, enemy['pos'].y - enemy['size']/2, enemy['size'], enemy['size'])
                if enemy_rect.colliderect(arrow_rect):
                    # SFX: Arrow Hit
                    reward += 0.1 # Base reward for any hit
                    self.combo_counter += 1
                    if self.combo_counter >= 3 and self.combo_counter % 3 == 0:
                        reward += 5.0 # Combo reward

                    damage = 10 # Base damage
                    is_correct_letter = False
                    try:
                        # Find first non-hit occurrence of the letter
                        idx = [i for i, char in enumerate(enemy['word']) if char == arrow['letter'] and not enemy['hit_letters'][i]][0]
                        enemy['hit_letters'][idx] = True
                        damage = 40 # Bonus damage for correct letter
                        reward += 0.5
                        is_correct_letter = True
                    except IndexError:
                        pass # Letter not in word or already hit

                    enemy['health'] -= damage
                    self._create_particles(arrow['pos'], self.COLOR_PARTICLE_GOLD, 20 if is_correct_letter else 5)
                    
                    if arrow in self.arrows:
                        self.arrows.remove(arrow)

                    if enemy['health'] <= 0:
                        reward += 2.0 # Reward for defeating enemy
                        self.enemies.remove(enemy)
                        # SFX: Enemy Defeated
                    break # Arrow can only hit one enemy
        return reward

    def _check_termination(self):
        if self.player_base_health <= 0:
            self.game_over = True
        if self.territories_conquered >= self.TOTAL_TERRITORIES:
            self.game_over = True
        return self.game_over
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_base_health,
            "territories_conquered": self.territories_conquered,
            "combo": self.combo_counter
        }

    def _render_game(self):
        # Render conquered territory indicators
        for i in range(self.TOTAL_TERRITORIES):
            color = self.COLOR_UI_ACCENT if i < self.territories_conquered else self.COLOR_BASE
            pygame.draw.rect(self.screen, color, (self.SCREEN_WIDTH - (self.TOTAL_TERRITORIES - i) * 15, 5, 12, 12))

        # Render player base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (0, 0, 80, self.SCREEN_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_BASE_STROKE, (80, 0), (80, self.SCREEN_HEIGHT), 2)
        
        # Render enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], enemy['size'], self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], enemy['size'], self.COLOR_ENEMY)
            
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (pos[0] - 20, pos[1] - 35, 40, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (pos[0] - 20, pos[1] - 35, 40 * health_ratio, 5))

            # Word target
            word_str = "".join(enemy['word'])
            word_surf = self.font_medium.render(word_str, True, self.COLOR_ENEMY_WORD)
            self.screen.blit(word_surf, (pos[0] - word_surf.get_width() / 2, pos[1] - 55))
            for i, char_hit in enumerate(enemy['hit_letters']):
                if char_hit:
                    char_surf = self.font_medium.render(enemy['word'][i], True, self.COLOR_ENEMY_WORD_HIT)
                    char_width_total = word_surf.get_width()
                    char_width_before = self.font_medium.render("".join(enemy['word'][:i]), True, (0,0,0)).get_width()
                    self.screen.blit(char_surf, (pos[0] - char_width_total / 2 + char_width_before, pos[1] - 55))


        # Render arrows
        for arrow in self.arrows:
            p1 = arrow['pos']
            angle_rad = math.radians(arrow['angle'])
            p2 = p1 - pygame.Vector2(20 * math.cos(angle_rad), 20 * math.sin(angle_rad))
            pygame.draw.line(self.screen, self.COLOR_ARROW, (p1.x, p1.y), (p2.x, p2.y), 3)
            letter_surf = self.font_small.render(arrow['letter'], True, self.COLOR_ARROW_LETTER)
            pygame.gfxdraw.filled_circle(self.screen, int(p1.x), int(p1.y), 5, self.COLOR_ARROW)
            self.screen.blit(letter_surf, letter_surf.get_rect(center=(int(p1.x), int(p1.y))))

        # Render player bow
        bow_pos = (80, self.SCREEN_HEIGHT / 2)
        angle_rad = math.radians(self.aim_angle)
        bow_tip = (bow_pos[0] + 30 * math.cos(angle_rad), bow_pos[1] + 30 * math.sin(angle_rad))
        pygame.draw.line(self.screen, self.COLOR_PLAYER, bow_pos, bow_tip, 4)
        pygame.gfxdraw.filled_circle(self.screen, int(bow_pos[0]), int(bow_pos[1]), 8, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(bow_pos[0]), int(bow_pos[1]), 8, self.COLOR_PLAYER)

        # Render particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(max(0, p['radius'])), p['color'])

    def _render_ui(self):
        # Player base health
        health_ratio = max(0, self.player_base_health / self.max_base_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (10, self.SCREEN_HEIGHT - 30, 60, 20))
        if health_ratio > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (10, self.SCREEN_HEIGHT - 30, 60 * health_ratio, 20))
        health_text = self.font_small.render(f"{int(self.player_base_health)}/{self.max_base_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, self.SCREEN_HEIGHT - 28))

        # Score
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Combo counter
        if self.combo_counter > 1:
            combo_text = self.font_large.render(f"{self.combo_counter}x COMBO!", True, self.COLOR_UI_ACCENT)
            self.screen.blit(combo_text, combo_text.get_rect(center=(self.SCREEN_WIDTH/2, 40)))

        # Letter inventory
        for i, letter in enumerate(self.player_letters):
            color = self.COLOR_UI_ACCENT if i == self.selected_letter_index else self.COLOR_UI_TEXT
            is_selected = i == self.selected_letter_index
            size = 28 if is_selected else 20
            font = self.font_medium if is_selected else self.font_small
            y_pos = self.SCREEN_HEIGHT / 2 - (len(self.player_letters) * 25 / 2) + i * 25
            
            letter_surf = font.render(letter, True, self.COLOR_BG)
            rect = pygame.Rect(0, 0, size, size)
            rect.center = (40, y_pos)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            self.screen.blit(letter_surf, letter_surf.get_rect(center=rect.center))

    def _create_particles(self, position, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': position.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'lifetime': self.np_random.integers(15, 31),
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not work with the "dummy" video driver, so we unset it.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Assyrian Archery Alphabet")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # ARROWS: Aim
    # SPACE: Shoot
    # SHIFT: Cycle Letter
    
    while not (terminated or truncated):
        # Default action is "do nothing"
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Territorries Conquered: {info['territories_conquered']}")
            
    env.close()