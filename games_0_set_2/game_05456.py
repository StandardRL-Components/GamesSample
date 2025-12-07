import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold shift for a speed boost. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a powerful robot in a top-down arena, blasting waves of enemy robots to achieve total robotic domination."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30

    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (200, 200, 200)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_PROJ_PLAYER = (255, 255, 0)
    COLOR_PROJ_ENEMY = (255, 100, 0)
    COLOR_HEALTH = (0, 255, 0)
    COLOR_HEALTH_BG = (100, 0, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_EXPLOSION = (255, 255, 255)
    
    PLAYER_SIZE = 16
    PLAYER_SPEED = 4
    PLAYER_HEALTH_MAX = 100
    PLAYER_SHOOT_COOLDOWN = 6 # frames
    PLAYER_BOOST_SPEED_MULTIPLIER = 2
    PLAYER_BOOST_DURATION = int(2 * FPS)
    PLAYER_BOOST_COOLDOWN = int(5 * FPS)

    ENEMY_SIZE = 14
    ENEMY_SPEED = 1.5
    ENEMIES_PER_WAVE = 20
    ENEMY_AI_TICK = 15 # frames

    PROJ_SPEED = 10
    PROJ_SIZE = 3

    MAX_STAGES = 3
    STAGE_TIME_SECONDS = 60
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_aim_direction = None
        self.player_shoot_cooldown_timer = None
        self.player_speed_boost_timer = None
        self.player_speed_boost_cooldown_timer = None
        self.enemies = []
        self.projectiles_player = []
        self.projectiles_enemy = []
        self.particles = []
        self.score = 0
        self.stage = 1
        self.stage_timer = 0
        self.stage_clear_timer = 0
        self.enemy_fire_rate_per_sec = 0
        self.game_over = False
        self.win_game = False
        self.steps = 0
        self.last_pos_for_stuck_check = None
        self.stuck_timer = 0
        
        # self.reset() # This is called by the validation function
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_aim_direction = pygame.Vector2(0, -1) # Default aim up
        self.player_shoot_cooldown_timer = 0
        self.player_speed_boost_timer = 0
        self.player_speed_boost_cooldown_timer = 0
        
        self.enemies = []
        self.projectiles_player = []
        self.projectiles_enemy = []
        self.particles = []
        
        self.score = 0
        self.stage = 1
        self.enemy_fire_rate_per_sec = 0.2
        self.game_over = False
        self.win_game = False
        self.steps = 0
        
        self.stage_timer = self.STAGE_TIME_SECONDS * self.FPS
        self.stage_clear_timer = 0
        
        self.last_pos_for_stuck_check = pygame.Vector2(self.player_pos)
        self.stuck_timer = 0

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def _spawn_wave(self):
        self.enemies = []
        for _ in range(self.ENEMIES_PER_WAVE):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(self.ENEMY_SIZE, self.WIDTH - self.ENEMY_SIZE),
                    self.np_random.uniform(self.ENEMY_SIZE, self.HEIGHT - self.ENEMY_SIZE)
                )
                if pos.distance_to(self.player_pos) > 100: # Don't spawn on player
                    # Check for overlap with other enemies
                    if not any(pos.distance_to(e['pos']) < self.ENEMY_SIZE * 2 for e in self.enemies):
                        break
            
            self.enemies.append({
                'pos': pos,
                'state': 'PATROL',
                'ai_timer': self.np_random.integers(0, self.ENEMY_AI_TICK),
                'patrol_target': pygame.Vector2(pos)
            })

    def step(self, action):
        reward = 0
        
        # Handle stage clear intermission
        if self.stage_clear_timer > 0:
            self.stage_clear_timer -= 1
            if self.stage_clear_timer == 0:
                self.stage += 1
                if self.stage > self.MAX_STAGES:
                    self.win_game = True
                else:
                    self.stage_timer = self.STAGE_TIME_SECONDS * self.FPS
                    self.enemy_fire_rate_per_sec += 0.02
                    self._spawn_wave()
            
            terminated = self.win_game
            return self._get_observation(), reward, terminated, False, self._get_info()

        # If game is over, do nothing
        if self.game_over or self.win_game:
            terminated = self.game_over or self.win_game
            return self._get_observation(), 0, terminated, False, self._get_info()

        self.steps += 1
        self.stage_timer -= 1

        # --- UPDATE TIMERS ---
        if self.player_shoot_cooldown_timer > 0: self.player_shoot_cooldown_timer -= 1
        if self.player_speed_boost_timer > 0: self.player_speed_boost_timer -= 1
        if self.player_speed_boost_cooldown_timer > 0: self.player_speed_boost_cooldown_timer -= 1
        
        # --- UNPACK ACTIONS ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- PLAYER LOGIC ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1

        # Reward for moving towards/away from nearest enemy
        dist_before = float('inf')
        if self.enemies:
            nearest_enemy = min(self.enemies, key=lambda e: self.player_pos.distance_to(e['pos']))
            dist_before = self.player_pos.distance_to(nearest_enemy['pos'])

        current_speed = self.PLAYER_SPEED
        if self.player_speed_boost_timer > 0:
            current_speed *= self.PLAYER_BOOST_SPEED_MULTIPLIER

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * current_speed
            self.player_aim_direction = pygame.Vector2(move_vec)
        
        if self.enemies:
            dist_after = self.player_pos.distance_to(nearest_enemy['pos'])
            if dist_after < dist_before: reward += 0.01
            else: reward -= 0.01

        # Anti-softlock check
        self.stuck_timer += 1
        if self.stuck_timer >= self.FPS * 5:
            if self.player_pos.distance_to(self.last_pos_for_stuck_check) < 10:
                self.player_pos = pygame.Vector2(
                    self.np_random.uniform(self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE),
                    self.np_random.uniform(self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)
                )
            self.last_pos_for_stuck_check = pygame.Vector2(self.player_pos)
            self.stuck_timer = 0

        # Player bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE/2, self.WIDTH - self.PLAYER_SIZE/2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE/2, self.HEIGHT - self.PLAYER_SIZE/2)

        # Player actions
        if space_held and self.player_shoot_cooldown_timer == 0:
            # sfx: player_shoot.wav
            self.projectiles_player.append({
                'pos': pygame.Vector2(self.player_pos),
                'vel': self.player_aim_direction * self.PROJ_SPEED
            })
            self.player_shoot_cooldown_timer = self.PLAYER_SHOOT_COOLDOWN
        
        if shift_held and self.player_speed_boost_cooldown_timer == 0 and self.player_speed_boost_timer == 0:
            # sfx: boost.wav
            self.player_speed_boost_timer = self.PLAYER_BOOST_DURATION
            self.player_speed_boost_cooldown_timer = self.PLAYER_BOOST_COOLDOWN

        # --- ENEMY LOGIC ---
        for enemy in self.enemies:
            enemy['ai_timer'] -= 1
            player_dist = enemy['pos'].distance_to(self.player_pos)

            # FSM State Transition
            if enemy['ai_timer'] <= 0:
                enemy['ai_timer'] = self.ENEMY_AI_TICK + self.np_random.integers(-5, 5)
                # Evade logic
                is_danger = False
                for p in self.projectiles_player:
                    if p['pos'].distance_to(enemy['pos']) < 60:
                        is_danger = True; break
                
                if is_danger:
                    enemy['state'] = 'EVADE'
                elif player_dist < 200:
                    enemy['state'] = 'ATTACK'
                else:
                    enemy['state'] = 'PATROL'
            
            # FSM Action
            if enemy['state'] == 'PATROL':
                if enemy['pos'].distance_to(enemy['patrol_target']) < self.ENEMY_SPEED:
                    enemy['patrol_target'] = pygame.Vector2(
                        self.np_random.uniform(self.ENEMY_SIZE, self.WIDTH - self.ENEMY_SIZE),
                        self.np_random.uniform(self.ENEMY_SIZE, self.HEIGHT - self.ENEMY_SIZE)
                    )
                direction = (enemy['patrol_target'] - enemy['pos']).normalize()
                enemy['pos'] += direction * self.ENEMY_SPEED
            
            elif enemy['state'] == 'ATTACK':
                direction = (self.player_pos - enemy['pos'])
                if direction.length() > 0:
                    direction.normalize_ip()
                    enemy['pos'] += direction * self.ENEMY_SPEED * 0.5 # Slower when attacking
                    
                    # Fire projectile
                    if self.np_random.random() < (self.enemy_fire_rate_per_sec / self.FPS):
                        # sfx: enemy_shoot.wav
                        self.projectiles_enemy.append({
                            'pos': pygame.Vector2(enemy['pos']),
                            'vel': direction * self.PROJ_SPEED * 0.7
                        })

            elif enemy['state'] == 'EVADE':
                closest_proj = min(self.projectiles_player, key=lambda p: p['pos'].distance_to(enemy['pos']), default=None)
                if closest_proj:
                    evade_dir = (enemy['pos'] - closest_proj['pos']).normalize()
                    # Move perpendicular
                    evade_dir = evade_dir.rotate(90)
                    enemy['pos'] += evade_dir * self.ENEMY_SPEED * 1.2
            
            # Enemy bounds
            enemy['pos'].x = np.clip(enemy['pos'].x, self.ENEMY_SIZE/2, self.WIDTH - self.ENEMY_SIZE/2)
            enemy['pos'].y = np.clip(enemy['pos'].y, self.ENEMY_SIZE/2, self.HEIGHT - self.ENEMY_SIZE/2)

        # --- PROJECTILE & COLLISION LOGIC ---
        # Player projectiles
        for p in self.projectiles_player[:]:
            p['pos'] += p['vel']
            if not (0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT):
                self.projectiles_player.remove(p)
                continue
            
            for enemy in self.enemies[:]:
                if enemy['pos'].distance_to(p['pos']) < (self.ENEMY_SIZE + self.PROJ_SIZE) / 2:
                    # sfx: explosion.wav
                    for _ in range(20):
                        self.particles.append(self._create_particle(enemy['pos']))
                    self.enemies.remove(enemy)
                    self.projectiles_player.remove(p)
                    self.score += 1
                    reward += 1
                    break
        
        # Enemy projectiles
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for p in self.projectiles_enemy[:]:
            p['pos'] += p['vel']
            if not (0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT):
                self.projectiles_enemy.remove(p)
                continue
            
            if player_rect.collidepoint(p['pos']):
                # sfx: player_hit.wav
                self.projectiles_enemy.remove(p)
                self.player_health -= 10
                reward -= 1
                break

        # --- PARTICLE LOGIC ---
        for part in self.particles[:]:
            part['life'] -= 1
            if part['life'] <= 0:
                self.particles.remove(part)
            else:
                part['pos'] += part['vel']
                part['radius'] = (part['life'] / part['max_life']) * part['max_radius']

        # --- TERMINATION & STAGE LOGIC ---
        if self.player_health <= 0:
            self.game_over = True
            reward -= 100
        
        if self.stage_timer <= 0:
            self.game_over = True
            reward -= 100 # Penalty for running out of time

        if not self.enemies and not self.win_game:
            self.stage_clear_timer = 2 * self.FPS
            reward += 10
            if self.stage == self.MAX_STAGES:
                reward += 100 # Add win bonus at the moment of clearing final stage
                
        terminated = self.game_over or self.win_game

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particle(self, pos):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        max_life = self.np_random.integers(15, 30)
        return {
            'pos': pygame.Vector2(pos),
            'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
            'life': max_life,
            'max_life': max_life,
            'max_radius': self.np_random.uniform(2, 5)
        }

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Particles
        for part in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(part['pos'].x), int(part['pos'].y),
                int(part['radius']), self.COLOR_EXPLOSION
            )

        # Enemy Projectiles
        for p in self.projectiles_enemy:
            pygame.draw.line(self.screen, self.COLOR_PROJ_ENEMY, p['pos'], p['pos'] - p['vel']/2, self.PROJ_SIZE)

        # Player Projectiles
        for p in self.projectiles_player:
            pygame.draw.line(self.screen, self.COLOR_PROJ_PLAYER, p['pos'], p['pos'] - p['vel']/2, self.PROJ_SIZE)
        
        # Enemies
        for enemy in self.enemies:
            p1 = (enemy['pos'].x, enemy['pos'].y - self.ENEMY_SIZE/2)
            p2 = (enemy['pos'].x - self.ENEMY_SIZE/2, enemy['pos'].y + self.ENEMY_SIZE/2)
            p3 = (enemy['pos'].x + self.ENEMY_SIZE/2, enemy['pos'].y + self.ENEMY_SIZE/2)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_ENEMY)

        # Player
        if self.player_pos is not None:
            player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
            player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
            
            # Glow effect
            glow_radius = self.PLAYER_SIZE * 1.5
            if self.player_speed_boost_timer > 0:
                glow_radius *= 1 + (self.player_speed_boost_timer / self.PLAYER_BOOST_DURATION)
            pygame.gfxdraw.filled_circle(self.screen, player_rect.centerx, player_rect.centery, int(glow_radius), self.COLOR_PLAYER_GLOW)
            
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        # Health Bar
        health_ratio = 0
        if self.player_health is not None:
            health_ratio = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, 200 * health_ratio, 20))

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Stage
        stage_text = self.font_main.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH/2 - stage_text.get_width()/2, 10))

        # Timer
        time_left = max(0, self.stage_timer / self.FPS)
        timer_text = self.font_main.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH/2 - timer_text.get_width()/2, 30))

        # Stage Clear Message
        if self.stage_clear_timer > 0 and self.stage < self.MAX_STAGES:
            clear_text = self.font_medium.render("STAGE CLEAR", True, self.COLOR_TEXT)
            self.screen.blit(clear_text, (self.WIDTH/2 - clear_text.get_width()/2, self.HEIGHT/2 - 50))
        
        # Game Over / Win Message
        if self.game_over:
            end_text = self.font_big.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - 30))
        elif self.win_game:
            win_text = self.font_big.render("YOU WIN!", True, self.COLOR_PLAYER)
            self.screen.blit(win_text, (self.WIDTH/2 - win_text.get_width()/2, self.HEIGHT/2 - 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "health": self.player_health,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and visualize the game
if __name__ == '__main__':
    # Un-comment the next line to run with a display window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window setup for visualization ---
    pygame.display.set_caption("Robot Arena")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        
        # Map keyboard inputs to actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)
        
    env.close()