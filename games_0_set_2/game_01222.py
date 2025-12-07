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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to fire your weapon. Good luck, pilot!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade shooter. Defend your sector from waves of alien invaders."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_THRUSTER = (255, 100, 0)
        self.COLOR_P_PROJ = (255, 255, 255)
        self.COLOR_A_PROJ = (255, 50, 50)
        self.COLOR_ALIEN_RED = (255, 60, 60)
        self.COLOR_ALIEN_BLUE = (60, 120, 255)
        self.COLOR_ALIEN_YELLOW = (255, 255, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_EXPLOSION = [(255, 100, 0), (255, 200, 0), (255, 255, 255)]

        # Game constants
        self.PLAYER_SPEED = 6
        self.PLAYER_FIRE_COOLDOWN_MAX = 8 # frames
        self.PROJECTILE_SPEED = 10
        self.MAX_STAGES = 3
        self.ALIENS_PER_STAGE = 50
        self.ALIENS_PER_WAVE = 10
        self.MAX_EPISODE_STEPS = 2000

        # Initialize state variables
        self.player_pos = None
        self.player_lives = None
        self.player_fire_cooldown = None
        self.player_projectiles = None
        self.aliens = None
        self.alien_projectiles = None
        self.alien_formation_dir = None
        self.alien_formation_move_cooldown = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.stage = None
        self.aliens_destroyed_total = None
        self.aliens_destroyed_in_wave = None
        
        # This is called in __init__ to ensure the np_random generator is available
        # but the actual game state reset happens in the public reset() method.
        # self.reset() 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.stage = 1
        self.aliens_destroyed_total = 0
        self.aliens_destroyed_in_wave = 0
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_lives = 3
        self.player_fire_cooldown = 0

        self.player_projectiles = []
        self.alien_projectiles = []
        self.aliens = []
        self.particles = []
        
        self.alien_formation_dir = 1
        self.alien_formation_move_cooldown = 0
        
        self._spawn_stars()
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for time passing
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1  # Boolean
            # shift_held = action[2] == 1 # Unused in this game

            self._handle_input(movement, space_held)
            self._update_player()
            self._update_projectiles()
            self._update_aliens()
            self._update_particles()
            
            reward += self._handle_collisions()
            
            if self.aliens_destroyed_in_wave >= self.ALIENS_PER_WAVE:
                self.aliens_destroyed_in_wave = 0
                reward += 2 # Wave clear bonus
                if self.aliens_destroyed_total < self.ALIENS_PER_STAGE * self.stage:
                    # SFX: Wave clear success
                    self._spawn_wave()
                elif self.stage < self.MAX_STAGES:
                    self.stage += 1
                    # SFX: Stage clear success
                    self._spawn_wave()
                else: # Game won
                    self.game_won = True

        self.steps += 1
        terminated, term_reward = self._check_termination()
        reward += term_reward
        
        if terminated:
            self.game_over = True
        
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, if truncated is true, terminated must be true

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Player Movement
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED

        # Player Firing
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        
        if space_held and self.player_fire_cooldown == 0:
            # SFX: Player shoot
            self.player_projectiles.append(pygame.Rect(self.player_pos[0] - 2, self.player_pos[1] - 20, 4, 15))
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX

    def _update_player(self):
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], self.HEIGHT - 100, self.HEIGHT - 20)

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if p.y > -p.height]
        for p in self.player_projectiles:
            p.y -= self.PROJECTILE_SPEED
            
        self.alien_projectiles = [p for p in self.alien_projectiles if p['rect'].y < self.HEIGHT]
        for p in self.alien_projectiles:
            p['rect'].y += p['speed']

    def _update_aliens(self):
        if not self.aliens:
            return

        # Movement
        move_sideways = False
        move_down = False
        
        # Difficulty scaling
        difficulty_tier = self.aliens_destroyed_total // 10
        move_speed = 10 + difficulty_tier * 2
        
        self.alien_formation_move_cooldown -= 1
        if self.alien_formation_move_cooldown <= 0:
            move_sideways = True
            self.alien_formation_move_cooldown = max(5, 30 - difficulty_tier)

        min_x = min(a['rect'].x for a in self.aliens)
        max_x = max(a['rect'].right for a in self.aliens)

        if (self.alien_formation_dir == 1 and max_x > self.WIDTH - 20) or \
           (self.alien_formation_dir == -1 and min_x < 20):
            self.alien_formation_dir *= -1
            move_down = True

        for alien in self.aliens:
            if move_sideways:
                alien['rect'].x += move_speed * self.alien_formation_dir
            if move_down:
                alien['rect'].y += 20
                if alien['rect'].bottom > self.HEIGHT - 100:
                    self.game_over = True # Aliens reached player zone
        
        # Firing
        fire_rate = 0.002 + difficulty_tier * 0.001
        for alien in self.aliens:
            if self.np_random.random() < fire_rate:
                # Check if an alien is clear to fire (no other aliens below it)
                can_fire = True
                for other_alien in self.aliens:
                    if alien is not other_alien and \
                       abs(alien['rect'].centerx - other_alien['rect'].centerx) < 10 and \
                       other_alien['rect'].y > alien['rect'].y:
                        can_fire = False
                        break
                if can_fire:
                    # SFX: Alien shoot
                    speed = 4 + difficulty_tier * 0.2
                    proj_rect = pygame.Rect(alien['rect'].centerx - 2, alien['rect'].bottom, 5, 10)
                    self.alien_projectiles.append({'rect': proj_rect, 'speed': speed})

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - 15, self.player_pos[1] - 10, 30, 20)

        # Player projectiles vs Aliens
        aliens_hit_indices = set()
        for i, proj in enumerate(self.player_projectiles):
            for j, alien in enumerate(self.aliens):
                if proj.colliderect(alien['rect']):
                    aliens_hit_indices.add(j)
                    self.player_projectiles[i] = None # Mark for removal
                    break
        
        if aliens_hit_indices:
            # SFX: Explosion
            for i in sorted(list(aliens_hit_indices), reverse=True):
                self._create_explosion(self.aliens[i]['rect'].center, 20)
                del self.aliens[i]
                self.score += 1
                reward += 1
                self.aliens_destroyed_total += 1
                self.aliens_destroyed_in_wave += 1
            self.player_projectiles = [p for p in self.player_projectiles if p is not None]

        # Alien projectiles vs Player
        for i, proj in enumerate(self.alien_projectiles):
            if player_rect.colliderect(proj['rect']):
                # SFX: Player hit
                self.alien_projectiles[i] = None # Mark for removal
                self.player_lives -= 1
                reward -= 1
                self._create_explosion(self.player_pos, 30)
                if self.player_lives > 0:
                    self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50] # Reset position
                break
        self.alien_projectiles = [p for p in self.alien_projectiles if p is not None]

        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.2

    def _check_termination(self):
        terminated = False
        reward = 0
        
        if self.game_won:
            terminated = True
            reward = 100
        elif self.player_lives <= 0:
            terminated = True
            reward = -100
        elif self.game_over: # Aliens reached the bottom
            terminated = True
            reward = -50
        
        return terminated, reward

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "stage": self.stage,
            "aliens_left": len(self.aliens)
        }

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for star in self.stars:
            # Parallax effect
            star['pos'][1] = (star['pos'][1] + star['speed']) % self.HEIGHT
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['radius'])

    def _render_game(self):
        self._render_particles()
        self._render_projectiles()
        self._render_aliens()
        if self.player_lives > 0:
            self._render_player()

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        # Ship body
        points = [(x, y - 12), (x - 15, y + 10), (x + 15, y + 10)]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        # Cockpit
        pygame.draw.circle(self.screen, self.COLOR_BG, (x, y), 5)
        pygame.gfxdraw.aacircle(self.screen, x, y, 5, self.COLOR_PLAYER)
        # Thruster
        thruster_y = y + 12 + random.randint(0, 5)
        thruster_points = [(x-5, y+10), (x+5, y+10), (x, thruster_y)]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER_THRUSTER, thruster_points)

    def _render_aliens(self):
        for alien in self.aliens:
            color = self.COLOR_ALIEN_RED
            if alien['type'] == 'blue':
                color = self.COLOR_ALIEN_BLUE
            elif alien['type'] == 'yellow':
                color = self.COLOR_ALIEN_YELLOW

            if alien['type'] == 'red': # Square
                pygame.draw.rect(self.screen, color, alien['rect'], border_radius=3)
            elif alien['type'] == 'blue': # Diamond
                p = alien['rect']
                points = [p.midtop, p.midright, p.midbottom, p.midleft]
                pygame.draw.polygon(self.screen, color, points)
                pygame.gfxdraw.aapolygon(self.screen, points, color)
            elif alien['type'] == 'yellow': # Hexagon
                p = alien['rect']
                r = p.width / 2
                points = [(p.centerx + r * math.cos(math.radians(a)), p.centery + r * math.sin(math.radians(a))) for a in range(30, 361, 60)]
                pygame.draw.polygon(self.screen, color, points)
                pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_projectiles(self):
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_P_PROJ, p, border_radius=2)
        for p in self.alien_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_A_PROJ, p['rect'], border_radius=2)
    
    def _render_particles(self):
        for p in self.particles:
            if p['radius'] > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Stage
        stage_text = self.font_medium.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 120, 10))
        
        # Lives
        lives_text = self.font_medium.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 100, 10))
        for i in range(self.player_lives):
            x, y = self.WIDTH - 40 + (i * 15), 25
            points = [(x, y - 6), (x - 7, y + 5), (x + 7, y + 5)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

        if self.game_over:
            msg = "GAME OVER"
            if self.game_won:
                msg = "VICTORY!"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _spawn_wave(self):
        rows = 2
        cols = self.ALIENS_PER_WAVE // rows
        start_x = self.WIDTH // 2 - (cols * 40) // 2
        start_y = 50
        
        for r in range(rows):
            for c in range(cols):
                alien_type = 'red'
                if self.stage >= 2 and self.np_random.random() < 0.4:
                    alien_type = 'blue'
                if self.stage >= 3 and self.np_random.random() < 0.2:
                    alien_type = 'yellow'
                
                x = start_x + c * 40
                y = start_y + r * 40
                self.aliens.append({
                    'rect': pygame.Rect(x, y, 30, 30),
                    'type': alien_type,
                })

    def _spawn_stars(self):
        self.stars = []
        for _ in range(150):
            speed = self.np_random.random() * 0.5 + 0.1
            self.stars.append({
                'pos': [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                'radius': self.np_random.random() * 1.5,
                'color': (c := int(100 + speed * 200), c, c),
                'speed': speed
            })

    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 4 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'radius': self.np_random.random() * 4 + 2,
                'color': random.choice(self.COLOR_EXPLOSION)
            })

    def render(self):
        return self._get_observation()
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To do so, you might need to unset the dummy video driver
    # comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # and install the full pygame library: pip install pygame
    
    # For testing purposes, we'll keep it headless
    
    def test_episode(env, max_steps: int = 1000):
        """Test running a full episode"""
        try:
            obs, info = env.reset()
            total_reward = 0
            
            for i in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                
                # Validate observation remains in bounds
                if obs.min() < 0 or obs.max() > 255:
                    return False, f"Step {i}: Observation out of bounds"
                
                if terminated:
                    break
            
            return True, f"Episode completed: {i+1} steps, reward={total_reward:.2f}"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Episode test error: {str(e)}"

    env = GameEnv(render_mode="rgb_array")
    success, message = test_episode(env)
    print(f"Test Result: {'Success' if success else 'Failure'}")
    print(message)
    env.close()