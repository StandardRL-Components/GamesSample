
# Generated: 2025-08-27T12:59:03.210417
# Source Brief: brief_00222.md
# Brief Index: 222

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move your ship. Survive the alien horde to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape hordes of descending aliens and reach your mothership in this fast-paced, side-scrolling arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Sizing
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 20
    ALIEN_RADIUS = 8
    POWERUP_RADIUS = 10
    SPACESHIP_Y = 30

    # Colors (Vibrant Neon)
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_SHIELD = (255, 255, 255)
    COLOR_STAR = (200, 200, 255)
    COLOR_SPACESHIP = (180, 180, 200)
    ALIEN_COLORS = {
        'linear': (255, 50, 50),   # Red
        'sine':   (255, 150, 50),  # Orange
        'zigzag': (50, 255, 50),   # Green
        'fast':   (255, 255, 50),  # Yellow
        'slow':   (200, 50, 255),  # Purple
    }
    POWERUP_COLORS = {
        'shield': (255, 255, 255), # White
        'score': (255, 215, 0),   # Gold
    }
    COLOR_UI_TEXT = (220, 220, 220)

    # Gameplay
    PLAYER_SPEED = 5.0
    MAX_STEPS = 5000
    STAR_COUNT = 150
    INITIAL_ALIEN_SPAWN_RATE = 40 # Lower is faster
    INITIAL_POWERUP_SPAWN_RATE = 800
    INITIAL_ALIEN_SPEED = 1.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Internal state variables are initialized in reset()
        self.player_pos = None
        self.aliens = None
        self.powerups = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_movement_direction = 0
        self.alien_spawn_timer = None
        self.powerup_spawn_timer = None
        self.base_alien_speed = None
        self.active_powerups = None
        
        self.np_random = None # Will be initialized in reset
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40)
        self.aliens = []
        self.powerups = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_movement_direction = 0 # -1 for left, 1 for right, 0 for none
        
        self.alien_spawn_timer = self.INITIAL_ALIEN_SPAWN_RATE
        self.powerup_spawn_timer = self.INITIAL_POWERUP_SPAWN_RATE
        self.base_alien_speed = self.INITIAL_ALIEN_SPEED
        self.active_powerups = {} # e.g. {'shield': 150} for 150 frames

        if self.stars is None:
            self.stars = [
                (
                    (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                    random.uniform(0.5, 1.5)
                ) for _ in range(self.STAR_COUNT)
            ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        # space_held and shift_held are not used in this game as per the brief
        
        self.steps += 1
        reward = 0.1 # Survival reward

        self._handle_player_movement(movement)
        self._update_difficulty()
        self._spawn_entities()
        
        # These events must be processed before reward calculation
        dodged_aliens_count = self._update_aliens()
        collected_powerup = self._update_powerups()
        self._update_particles()
        self._update_active_powerups()
        
        # Calculate rewards based on events this frame
        reward += self._calculate_movement_reward()
        if collected_powerup:
            reward += 1.0 # sfx: powerup_get.wav
        reward += 5.0 * dodged_aliens_count

        terminated = self._check_collisions() or self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.steps >= self.MAX_STEPS:
                reward = 100.0 # Victory reward
            elif self.game_over:
                reward = -50.0 # Collision penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        self.player_movement_direction = 0
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
            self.player_movement_direction = -1
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
            self.player_movement_direction = 1
            
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_WIDTH / 2, self.SCREEN_WIDTH - self.PLAYER_WIDTH / 2)

    def _update_difficulty(self):
        # Increase alien speed every 500 steps
        if self.steps > 0 and self.steps % 500 == 0:
            self.base_alien_speed += 0.05
        # Decrease power-up spawn rate every 1000 steps (by increasing timer)
        if self.steps > 0 and self.steps % 1000 == 0:
            self.INITIAL_POWERUP_SPAWN_RATE = max(200, self.INITIAL_POWERUP_SPAWN_RATE * 1.01)

    def _spawn_entities(self):
        # Spawn Aliens
        self.alien_spawn_timer -= 1
        if self.alien_spawn_timer <= 0:
            alien_type = random.choice(list(self.ALIEN_COLORS.keys()))
            speed_multiplier = 1.0
            if alien_type == 'fast': speed_multiplier = 1.5
            if alien_type == 'slow': speed_multiplier = 0.7
            
            self.aliens.append({
                'pos': pygame.Vector2(random.randint(self.ALIEN_RADIUS, self.SCREEN_WIDTH - self.ALIEN_RADIUS), -self.ALIEN_RADIUS),
                'type': alien_type,
                'speed_mult': speed_multiplier,
                'trail': deque(maxlen=15),
                'phase': random.uniform(0, 2 * math.pi) # For sine/zigzag
            })
            self.alien_spawn_timer = int(self.INITIAL_ALIEN_SPAWN_RATE * random.uniform(0.8, 1.2))

        # Spawn Powerups
        self.powerup_spawn_timer -= 1
        if self.powerup_spawn_timer <= 0:
            powerup_type = random.choice(list(self.POWERUP_COLORS.keys()))
            self.powerups.append({
                'pos': pygame.Vector2(random.randint(self.POWERUP_RADIUS, self.SCREEN_WIDTH - self.POWERUP_RADIUS), -self.POWERUP_RADIUS),
                'type': powerup_type
            })
            self.powerup_spawn_timer = int(self.INITIAL_POWERUP_SPAWN_RATE * random.uniform(0.8, 1.5))
    
    def _update_aliens(self):
        dodged_aliens_count = 0
        for alien in self.aliens[:]:
            alien['trail'].append(alien['pos'].copy())
            speed = self.base_alien_speed * alien['speed_mult']
            
            if alien['type'] == 'sine':
                alien['pos'].x += math.sin(alien['phase'] + self.steps * 0.05) * 2.5
            elif alien['type'] == 'zigzag':
                alien['pos'].x += math.copysign(2.0, math.sin(alien['phase'] + self.steps * 0.05))

            alien['pos'].y += speed
            
            # Risky dodge check
            if not self.game_over and abs(alien['pos'].y - self.player_pos.y) < 2:
                if 0 < abs(alien['pos'].x - self.player_pos.x) < self.PLAYER_WIDTH:
                    dodged_aliens_count += 1
                    self.score += 10 # Bonus score for risky dodge

            if alien['pos'].y > self.SCREEN_HEIGHT + self.ALIEN_RADIUS:
                self.aliens.remove(alien)
                self.score += 1
        return dodged_aliens_count

    def _update_powerups(self):
        collected_powerup = False
        for powerup in self.powerups[:]:
            powerup['pos'].y += self.INITIAL_ALIEN_SPEED * 0.8
            if self.player_pos.distance_to(powerup['pos']) < self.PLAYER_WIDTH / 2 + self.POWERUP_RADIUS:
                self._activate_powerup(powerup['type'])
                self.powerups.remove(powerup)
                collected_powerup = True
            elif powerup['pos'].y > self.SCREEN_HEIGHT + self.POWERUP_RADIUS:
                self.powerups.remove(powerup)
        return collected_powerup
        
    def _activate_powerup(self, type):
        # sfx: powerup_activate.wav
        if type == 'shield':
            self.active_powerups['shield'] = 240 # 8 seconds at 30fps
        elif type == 'score':
            self.score += 100
        
        self._create_particles(self.player_pos, self.POWERUP_COLORS[type], 30)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_active_powerups(self):
        for effect in list(self.active_powerups.keys()):
            self.active_powerups[effect] -= 1
            if self.active_powerups[effect] <= 0:
                del self.active_powerups[effect]
    
    def _calculate_movement_reward(self):
        if self.player_movement_direction == 0 or not self.aliens:
            return 0
        
        closest_alien = min(self.aliens, key=lambda a: self.player_pos.distance_squared_to(a['pos']))
        dist_vector = closest_alien['pos'] - self.player_pos
        
        # If moving left (-1), and alien is to the right (dist_vector.x > 0), that's good.
        # If moving right (1), and alien is to the left (dist_vector.x < 0), that's good.
        # This is equivalent to sign(move) != sign(dist_vector.x)
        if np.sign(self.player_movement_direction) != np.sign(dist_vector.x):
            return 0.5 # Moving away
        else:
            return -0.2 # Moving towards
            
    def _check_collisions(self):
        if 'shield' in self.active_powerups:
            return False
            
        for alien in self.aliens:
            if self.player_pos.distance_to(alien['pos']) < self.PLAYER_WIDTH / 2 + self.ALIEN_RADIUS * 0.8:
                self.game_over = True
                # sfx: player_explosion.wav
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 50)
                self._create_particles(alien['pos'], self.ALIEN_COLORS[alien['type']], 50)
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "is_shielded": 'shield' in self.active_powerups,
            "aliens_on_screen": len(self.aliens),
        }

    def _render_background(self):
        for pos, radius in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, pos, radius)
        
        # Render spaceship goal
        ss_x, ss_y = self.SCREEN_WIDTH / 2, self.SPACESHIP_Y
        pygame.draw.polygon(self.screen, self.COLOR_SPACESHIP, [
            (ss_x, ss_y - 15), (ss_x - 50, ss_y + 15), (ss_x + 50, ss_y + 15)
        ])
        pygame.draw.rect(self.screen, self.COLOR_SPACESHIP, (ss_x - 25, ss_y + 15, 50, 10))

    def _render_game(self):
        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            p['color'] = (*p['base_color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, p['color'], (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        # Alien trails and aliens
        for alien in self.aliens:
            for i, pos in enumerate(alien['trail']):
                alpha = int(100 * (i / len(alien['trail'])))
                color = (*self.ALIEN_COLORS[alien['type']], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(self.ALIEN_RADIUS * 0.5), color)
            pygame.gfxdraw.aacircle(self.screen, int(alien['pos'].x), int(alien['pos'].y), self.ALIEN_RADIUS, self.ALIEN_COLORS[alien['type']])
            pygame.gfxdraw.filled_circle(self.screen, int(alien['pos'].x), int(alien['pos'].y), self.ALIEN_RADIUS, self.ALIEN_COLORS[alien['type']])

        # Powerups
        for powerup in self.powerups:
            color = self.POWERUP_COLORS[powerup['type']]
            # Glow effect
            s = pygame.Surface((self.POWERUP_RADIUS*4, self.POWERUP_RADIUS*4), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, 50), (self.POWERUP_RADIUS*2, self.POWERUP_RADIUS*2), self.POWERUP_RADIUS*2)
            self.screen.blit(s, (int(powerup['pos'].x - self.POWERUP_RADIUS*2), int(powerup['pos'].y - self.POWERUP_RADIUS*2)))
            # Core
            pygame.gfxdraw.aacircle(self.screen, int(powerup['pos'].x), int(powerup['pos'].y), self.POWERUP_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, int(powerup['pos'].x), int(powerup['pos'].y), self.POWERUP_RADIUS, color)

        # Player
        if not self.game_over:
            p_x, p_y = self.player_pos.x, self.player_pos.y
            player_points = [
                (p_x, p_y - self.PLAYER_HEIGHT / 2),
                (p_x - self.PLAYER_WIDTH / 2, p_y + self.PLAYER_HEIGHT / 2),
                (p_x + self.PLAYER_WIDTH / 2, p_y + self.PLAYER_HEIGHT / 2),
            ]
            
            if 'shield' in self.active_powerups:
                shield_alpha = min(255, 50 + int(205 * (self.active_powerups['shield'] / 60))) if self.active_powerups['shield'] < 60 else 255
                shield_radius = int(self.PLAYER_WIDTH * 0.8)
                s = pygame.Surface((shield_radius*2, shield_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_PLAYER_SHIELD, int(shield_alpha*0.3)), (shield_radius, shield_radius), shield_radius)
                self.screen.blit(s, (int(p_x - shield_radius), int(p_y - shield_radius)))
                pygame.gfxdraw.aacircle(self.screen, int(p_x), int(p_y), shield_radius, (*self.COLOR_PLAYER_SHIELD, shield_alpha))
            
            pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.small_font.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        
        if self.game_over:
            end_text = self.font.render("GAME OVER", True, (255, 50, 50))
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))
        elif self.steps >= self.MAX_STEPS:
            end_text = self.font.render("YOU ESCAPED!", True, (50, 255, 50))
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            life = random.randint(20, 50)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': life,
                'max_life': life,
                'base_color': color,
                'size': random.randint(1, 4)
            })

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Alien Escape")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(env.user_guide)
    print(env.game_description)

    while not terminated:
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # space and shift are not used

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        clock.tick(30) # Run at 30 FPS

    env.close()