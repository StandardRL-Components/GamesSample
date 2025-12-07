import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to attack in the corresponding direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of monsters in an isometric arena. Time your attacks to survive!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Pygame headless setup
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_DMG = (255, 100, 100)
        self.COLOR_MONSTER_A = (255, 120, 0)
        self.COLOR_MONSTER_B = (200, 80, 255)
        self.COLOR_MONSTER_C = (255, 50, 50)
        self.COLOR_MONSTER_DMG = (255, 255, 255)
        self.COLOR_HEALTH_BAR_BG = (80, 0, 0)
        self.COLOR_HEALTH_BAR_FG = (0, 200, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE_SLASH = (255, 255, 255)
        self.COLOR_PARTICLE_HIT = (255, 200, 0)
        self.COLOR_PARTICLE_DEATH = (150, 150, 150)

        # Isometric projection constants
        self.ISO_TILE_WIDTH = 48
        self.ISO_TILE_HEIGHT = 24
        self.GRID_SIZE = 10 # 10x10 grid
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 120

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # State variables (will be initialized in reset)
        self.rng = None
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.player_pos = None
        self.player_health = 0
        self.player_max_health = 0
        self.player_action_cooldown = 0
        self.player_iframes = 0 # Invincibility frames
        self.monsters = []
        self.particles = []
        
        # self.validate_implementation() # Commented out for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            if self.rng is None:
                 self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.wave = 1
        
        self.player_pos = np.array([0.0, 0.0]) # Grid coordinates
        self.player_max_health = 100
        self.player_health = self.player_max_health
        self.player_action_cooldown = 0
        self.player_iframes = 0

        self.monsters = []
        self.particles = []
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement_action = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.0
        terminated = False
        
        # Decrement timers
        self.player_action_cooldown = max(0, self.player_action_cooldown - 1)
        self.player_iframes = max(0, self.player_iframes - 1)
        
        # === Player Action ===
        if movement_action != 0 and self.player_action_cooldown == 0:
            # Action has a 0.2s cooldown (6 frames at 30fps)
            self.player_action_cooldown = 6 
            # sound: player_slash.wav
            reward += self._player_attack(movement_action)

        # === Monster AI and Update ===
        for monster in self.monsters[:]: # Iterate on a copy
            monster['ai_timer'] -= 1
            if monster['ai_timer'] <= 0:
                reward += self._update_monster_ai(monster)
        
        # === Update Particles ===
        for p in self.particles[:]:
            p['life'] -= 1
            if 'vel' in p:
                p['pos'] += p['vel']
            if p['life'] <= 0:
                self.particles.remove(p)

        # === Check for monster deaths ===
        for monster in self.monsters[:]:
            if monster['health'] <= 0:
                self.score += 100
                reward += 1.0 # Reward for defeating a monster
                self._create_death_particles(monster['pos'])
                self.monsters.remove(monster)
                # sound: monster_death.wav

        # === Check for wave clear ===
        if not self.monsters:
            self.score += 500 * self.wave
            reward += 100.0 # Large reward for clearing the wave
            terminated = True # Episode ends on wave clear

        # === Check for player death ===
        if self.player_health <= 0:
            reward -= 100.0
            terminated = True
        
        # === Step and time limit ===
        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # For many algos, truncated=True means terminated=True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.ISO_TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.ISO_TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _spawn_wave(self):
        self.monsters.clear()
        monster_health = 50 + (self.wave - 1) * 10
        
        spawn_points = [
            np.array([3.0, -3.0]), np.array([-3.0, 3.0]), np.array([3.0, 3.0]),
            np.array([-3.0, -3.0]), np.array([0.0, 4.0]), np.array([4.0, 0.0])
        ]
        self.rng.shuffle(spawn_points)

        for i in range(3):
            monster_type = self.rng.choice(['charger', 'shooter'])
            color = self.COLOR_MONSTER_A if monster_type == 'charger' else self.COLOR_MONSTER_B
            
            self.monsters.append({
                'pos': spawn_points[i],
                'health': monster_health,
                'max_health': monster_health,
                'type': monster_type,
                'color': color,
                'ai_state': 'idle',
                'ai_timer': self.rng.integers(30, 90),
                'target_pos': spawn_points[i].copy()
            })

    def _player_attack(self, direction):
        attack_reward = 0
        # 1=up, 2=down, 3=left, 4=right
        # Map to grid directions
        if direction == 1: attack_vec = np.array([0, -1]) # Up -> NW
        elif direction == 2: attack_vec = np.array([0, 1])  # Down -> SE
        elif direction == 3: attack_vec = np.array([-1, 0]) # Left -> SW
        else: attack_vec = np.array([1, 0]) # Right -> NE

        attack_pos = self.player_pos + attack_vec
        self._create_slash_particles(attack_pos, attack_vec)

        for monster in self.monsters:
            if np.linalg.norm(monster['pos'] - attack_pos) < 0.8:
                damage = 25
                monster['health'] -= damage
                self._create_hit_particles(monster['pos'])
                self._create_damage_number(monster['pos'], damage, self.COLOR_PLAYER)
                self.score += 10
                attack_reward += 0.1
        return attack_reward

    def _update_monster_ai(self, monster):
        reward = 0
        if monster['ai_state'] == 'idle':
            monster['ai_state'] = 'moving'
            # Move to a random spot within a radius of the player
            angle = self.rng.random() * 2 * math.pi
            radius = self.rng.uniform(2.5, 4.5)
            monster['target_pos'] = self.player_pos + np.array([math.cos(angle) * radius, math.sin(angle) * radius])
            monster['ai_timer'] = self.rng.integers(60, 120) # Time to move
        
        elif monster['ai_state'] == 'moving':
            # Move towards target
            move_vec = monster['target_pos'] - monster['pos']
            if np.linalg.norm(move_vec) > 0.1:
                monster['pos'] += move_vec / np.linalg.norm(move_vec) * 0.05 # Speed
            else:
                monster['ai_state'] = 'attacking'
                monster['ai_timer'] = 45 # Wind-up time
                monster['attack_telegraph_pos'] = self.player_pos.copy()

        elif monster['ai_state'] == 'attacking':
            # Attack is executed, now go back to idle
            # sound: monster_attack.wav
            if np.linalg.norm(self.player_pos - monster['attack_telegraph_pos']) < 1.0:
                if self.player_iframes == 0:
                    self.player_health -= 20
                    self.player_iframes = 30 # 1s of invincibility
                    self._create_hit_particles(self.player_pos, scale=1.5)
                    self._create_damage_number(self.player_pos, 20, self.COLOR_MONSTER_C)
                    reward -= 0.2
            
            monster['ai_state'] = 'idle'
            monster['ai_timer'] = self.rng.integers(30, 90)
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for i in range(-self.GRID_SIZE, self.GRID_SIZE + 1):
            # Horizontal lines
            p1 = self._iso_to_screen(i, -self.GRID_SIZE)
            p2 = self._iso_to_screen(i, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
            # Vertical lines
            p1 = self._iso_to_screen(-self.GRID_SIZE, i)
            p2 = self._iso_to_screen(self.GRID_SIZE, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        # Combine player and monsters to sort by Y for correct draw order
        entities = self.monsters + [{'type': 'player', 'pos': self.player_pos}]
        entities.sort(key=lambda e: e['pos'][0] + e['pos'][1])

        for entity in entities:
            if entity['type'] == 'player':
                self._render_player()
            else:
                self._render_monster(entity)
        
        # Render particles on top
        self._render_particles()

    def _render_player(self):
        screen_pos = self._iso_to_screen(self.player_pos[0], self.player_pos[1])
        color = self.COLOR_PLAYER if self.player_iframes % 4 < 2 else self.COLOR_PLAYER_DMG
        
        # Shadow
        pygame.gfxdraw.filled_ellipse(self.screen, screen_pos[0], screen_pos[1] + 18, 15, 8, (0,0,0,100))
        
        # Body
        body_rect = pygame.Rect(screen_pos[0] - 10, screen_pos[1] - 15, 20, 30)
        pygame.draw.rect(self.screen, color, body_rect, border_radius=5)
        
        # Health bar
        self._render_health_bar(screen_pos[0], screen_pos[1] - 25, self.player_health, self.player_max_health)

    def _render_monster(self, monster):
        screen_pos = self._iso_to_screen(monster['pos'][0], monster['pos'][1])
        color = monster['color']

        # Shadow
        pygame.gfxdraw.filled_ellipse(self.screen, screen_pos[0], screen_pos[1] + 16, 13, 7, (0,0,0,100))

        # Body
        body_rect = pygame.Rect(screen_pos[0] - 12, screen_pos[1] - 12, 24, 24)
        pygame.draw.rect(self.screen, color, body_rect, border_radius=8)

        # Health bar
        self._render_health_bar(screen_pos[0], screen_pos[1] - 20, monster['health'], monster['max_health'])

        # Attack telegraph
        if monster['ai_state'] == 'attacking':
            telegraph_pos = self._iso_to_screen(monster['attack_telegraph_pos'][0], monster['attack_telegraph_pos'][1])
            alpha = int(200 * (1 - monster['ai_timer'] / 45))
            pygame.gfxdraw.filled_circle(self.screen, telegraph_pos[0], telegraph_pos[1], 20, (255, 0, 0, alpha))
            pygame.gfxdraw.aacircle(self.screen, telegraph_pos[0], telegraph_pos[1], 20, (255, 100, 100, alpha))


    def _render_health_bar(self, x, y, health, max_health):
        width = 30
        height = 5
        health_ratio = max(0, health / max_health)
        bg_rect = pygame.Rect(x - width // 2, y, width, height)
        fg_rect = pygame.Rect(x - width // 2, y, int(width * health_ratio), height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, fg_rect)

    def _render_particles(self):
        for p in self.particles:
            pos = self._iso_to_screen(p['pos'][0], p['pos'][1])
            if p['type'] == 'slash':
                end_pos = self._iso_to_screen(p['end_pos'][0], p['end_pos'][1])
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*self.COLOR_PARTICLE_SLASH, alpha)
                # This requires a surface with per-pixel alpha
                line_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                pygame.draw.line(line_surf, color, pos, end_pos, 4)
                self.screen.blit(line_surf, (0,0))

            elif p['type'] == 'hit':
                size = int(p['scale'] * 15 * (p['life'] / p['max_life']))
                if size > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PARTICLE_HIT)
            
            elif p['type'] == 'death':
                size = int(20 * (1 - (p['life'] / p['max_life'])))
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*self.COLOR_PARTICLE_DEATH, alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            
            elif p['type'] == 'damage_number':
                alpha = int(255 * (p['life'] / p['max_life']))
                font_surface = self.font_small.render(str(p['text']), True, p['color'])
                font_surface.set_alpha(alpha)
                self.screen.blit(font_surface, (pos[0], pos[1] - 10))


    def _render_ui(self):
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        wave_text = self.font_small.render(f"Wave: {self.wave}", True, self.COLOR_TEXT)
        wave_rect = wave_text.get_rect(centerx=self.WIDTH // 2, y=10)
        self.screen.blit(wave_text, wave_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
        }

    def _create_slash_particles(self, pos, vec):
        self.particles.append({
            'type': 'slash',
            'pos': pos - vec * 0.5,
            'end_pos': pos + vec * 0.5,
            'life': 5,
            'max_life': 5
        })

    def _create_hit_particles(self, pos, scale=1.0):
        self.particles.append({
            'type': 'hit',
            'pos': pos,
            'life': 8,
            'max_life': 8,
            'scale': scale
        })

    def _create_death_particles(self, pos):
        for _ in range(10):
            self.particles.append({
                'type': 'death',
                'pos': pos,
                'life': 15,
                'max_life': 15
            })

    def _create_damage_number(self, pos, text, color):
        self.particles.append({
            'type': 'damage_number',
            'pos': pos,
            'vel': np.array([0, -0.05]), # Moves up in grid space
            'life': 20,
            'max_life': 20,
            'text': text,
            'color': color
        })
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # The environment is now headless by default
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    obs, info = env.reset()
    
    # To run with manual controls for testing:
    # 1. Comment out `os.environ["SDL_VIDEODRIVER"] = "dummy"` in __init__
    # 2. Uncomment the code below
    
    # real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Isometric Arena Fighter")
    # running = True
    # while running:
    #     movement = 0
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     if keys[pygame.K_DOWN]: movement = 2
    #     if keys[pygame.K_LEFT]: movement = 3
    #     if keys[pygame.K_RIGHT]: movement = 4
        
    #     action = [movement, 0, 0] # space/shift not used
    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}")
    #         obs, info = env.reset()
            
    #     # Draw the observation to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     real_screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
                
    #     env.clock.tick(env.FPS)
    
    # env.close()