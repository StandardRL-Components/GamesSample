
# Generated: 2025-08-28T04:17:57.598640
# Source Brief: brief_05202.md
# Brief Index: 5202

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to fire in your last direction of movement."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a robot through a neon arena, blasting waves of enemies to survive and progress through increasingly difficult stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        self.STAGES = 3

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 50, 50, 40)
        self.COLOR_PLAYER_PROJ = (100, 150, 255)
        self.COLOR_ENEMY_PROJ = (255, 150, 50)
        self.COLOR_WALL = (50, 50, 80)
        self.COLOR_UI = (220, 220, 255)
        self.COLOR_TEXT_POPUP = (255, 255, 100)
        self.COLOR_EXPLOSION = [(255, 255, 255), (255, 255, 100), (255, 150, 50)]

        # Player settings
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 4.0
        self.PLAYER_MAX_HEALTH = 50
        self.PLAYER_SHOOT_COOLDOWN = 6 # frames

        # Enemy settings
        self.ENEMY_SIZE = 10
        self.ENEMY_BASE_SPEED = 1.0
        self.ENEMY_MAX_HEALTH = 10
        self.ENEMY_SHOOT_COOLDOWN = 60 # frames
        self.BASE_ENEMY_COUNT = 8

        # Projectile settings
        self.PROJ_SPEED = 8.0
        self.PROJ_SIZE = 3

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_popup = pygame.font.Font(None, 20)
        self.font_main = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_last_move_dir = None
        self.player_shoot_cooldown_timer = None
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.floating_texts = []
        self.stage = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.stage_clear_timer = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.game_over = False
        self.stage_clear_timer = 0

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_last_move_dir = np.array([0, -1], dtype=np.float32) # Default up
        self.player_shoot_cooldown_timer = 0

        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.floating_texts = []
        
        self._spawn_enemies()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.clock.tick(30)
        
        reward = -0.01  # Small penalty for each step to encourage efficiency

        if not self.game_over:
            # Unpack factorized action
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            self._handle_player_input(movement, space_held)
            reward += self._update_entities()
            
            if self.stage_clear_timer > 0:
                self.stage_clear_timer -= 1
                if self.stage_clear_timer == 0:
                    if self.stage > self.STAGES:
                        self.game_over = True
                        reward += 100 # Final victory reward
                    else:
                        self._spawn_enemies()

            # Check for stage clear
            if not self.enemies and not self.game_over and self.stage_clear_timer == 0:
                self.stage += 1
                reward += 50 # Stage clear reward
                self.stage_clear_timer = 60 # 2-second pause

        # Check termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.player_health <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            reward = -100 # Game over penalty
            self._create_explosion(self.player_pos, 100, self.COLOR_EXPLOSION, 20)
            # sfx: player_explosion

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Update player movement
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1  # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1  # Right
        
        if np.linalg.norm(move_vec) > 0:
            self.player_last_move_dir = move_vec
            self.player_pos += move_vec * self.PLAYER_SPEED

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # Update shooting cooldown
        if self.player_shoot_cooldown_timer > 0:
            self.player_shoot_cooldown_timer -= 1
        
        # Handle shooting
        if space_held and self.player_shoot_cooldown_timer == 0:
            self.player_shoot_cooldown_timer = self.PLAYER_SHOOT_COOLDOWN
            proj_vel = self.player_last_move_dir * self.PROJ_SPEED
            self.projectiles.append({
                'pos': self.player_pos.copy(),
                'vel': proj_vel,
                'owner': 'player',
                'trail': []
            })
            # sfx: player_shoot

    def _update_entities(self):
        step_reward = 0
        
        # Update enemies
        enemy_projectiles_to_add = []
        for enemy in self.enemies:
            # Movement
            direction_to_player = self.player_pos - enemy['pos']
            dist = np.linalg.norm(direction_to_player)
            if dist > 0:
                enemy['pos'] += (direction_to_player / dist) * enemy['speed']
            
            # Shooting
            enemy['shoot_cooldown'] -= 1
            if enemy['shoot_cooldown'] <= 0:
                enemy['shoot_cooldown'] = self.ENEMY_SHOOT_COOLDOWN + self.np_random.integers(-10, 11)
                direction_to_player /= dist
                enemy_projectiles_to_add.append({
                    'pos': enemy['pos'].copy(),
                    'vel': direction_to_player * self.PROJ_SPEED * 0.75,
                    'owner': 'enemy',
                    'trail': []
                })
                # sfx: enemy_shoot
        self.projectiles.extend(enemy_projectiles_to_add)

        # Update projectiles and check collisions
        projectiles_to_keep = []
        for proj in self.projectiles:
            proj['pos'] += proj['vel']
            proj['trail'].append(proj['pos'].copy())
            if len(proj['trail']) > 5:
                proj['trail'].pop(0)

            # Boundary collision
            if not (0 < proj['pos'][0] < self.WIDTH and 0 < proj['pos'][1] < self.HEIGHT):
                continue # Remove projectile

            hit = False
            # Player projectile collision with enemies
            if proj['owner'] == 'player':
                for enemy in self.enemies:
                    if np.linalg.norm(proj['pos'] - enemy['pos']) < self.ENEMY_SIZE + self.PROJ_SIZE:
                        enemy['health'] -= 5
                        step_reward += 0.1 # Reward for hitting
                        self._create_explosion(proj['pos'], 5, [self.COLOR_ENEMY], 2)
                        hit = True
                        break
            # Enemy projectile collision with player
            elif proj['owner'] == 'enemy':
                if np.linalg.norm(proj['pos'] - self.player_pos) < self.PLAYER_SIZE + self.PROJ_SIZE:
                    self.player_health -= 5
                    self._create_explosion(proj['pos'], 10, [self.COLOR_PLAYER], 4)
                    # sfx: player_hit
                    hit = True

            if not hit:
                projectiles_to_keep.append(proj)
        self.projectiles = projectiles_to_keep

        # Check for defeated enemies
        enemies_to_keep = []
        for enemy in self.enemies:
            if enemy['health'] > 0:
                enemies_to_keep.append(enemy)
            else:
                step_reward += 10 # Reward for defeating
                self.score += 10
                self._create_explosion(enemy['pos'], 40, self.COLOR_EXPLOSION, 8)
                self._create_floating_text("+10", enemy['pos'], self.COLOR_TEXT_POPUP)
                # sfx: enemy_explosion
        self.enemies = enemies_to_keep

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] -= p['decay']

        # Update floating texts
        self.floating_texts = [ft for ft in self.floating_texts if ft['life'] > 0]
        for ft in self.floating_texts:
            ft['pos'][1] -= 0.5
            ft['life'] -= 1
        
        return step_reward

    def _spawn_enemies(self):
        enemy_count = self.BASE_ENEMY_COUNT + (self.stage - 1) * 2
        enemy_speed = self.ENEMY_BASE_SPEED + (self.stage - 1) * 0.15
        
        for _ in range(enemy_count):
            while True:
                pos = np.array([
                    self.np_random.uniform(self.ENEMY_SIZE, self.WIDTH - self.ENEMY_SIZE),
                    self.np_random.uniform(self.ENEMY_SIZE, self.HEIGHT - self.ENEMY_SIZE)
                ], dtype=np.float32)
                if np.linalg.norm(pos - self.player_pos) > 150: # Spawn away from player
                    break
            
            self.enemies.append({
                'pos': pos,
                'health': self.ENEMY_MAX_HEALTH,
                'shoot_cooldown': self.np_random.integers(1, self.ENEMY_SHOOT_COOLDOWN),
                'speed': enemy_speed * self.np_random.uniform(0.9, 1.1)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_walls()
        self._render_effects()
        self._render_projectiles()
        self._render_enemies()
        if self.player_health > 0:
            self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_walls(self):
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)

    def _render_player(self):
        # Glow
        glow_radius = int(self.PLAYER_SIZE * 2.5)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), glow_radius, self.COLOR_PLAYER_GLOW)

        # Body
        angle = math.atan2(self.player_last_move_dir[1], self.player_last_move_dir[0])
        points = []
        for i in range(3):
            a = angle + i * 2 * math.pi / 3
            p_angle = a + (math.pi/2 if i == 0 else -math.pi/6)
            radius = self.PLAYER_SIZE if i == 0 else self.PLAYER_SIZE * 0.7
            points.append((
                self.player_pos[0] + radius * math.cos(p_angle),
                self.player_pos[1] + radius * math.sin(p_angle)
            ))
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_PLAYER)

    def _render_enemies(self):
        for enemy in self.enemies:
            # Glow
            glow_radius = int(self.ENEMY_SIZE * 2.0)
            pygame.gfxdraw.filled_circle(self.screen, int(enemy['pos'][0]), int(enemy['pos'][1]), glow_radius, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.aacircle(self.screen, int(enemy['pos'][0]), int(enemy['pos'][1]), glow_radius, self.COLOR_ENEMY_GLOW)

            # Body (diamond shape)
            p = enemy['pos']
            s = self.ENEMY_SIZE
            points = [(p[0], p[1]-s), (p[0]+s, p[1]), (p[0], p[1]+s), (p[0]-s, p[1])]
            pygame.gfxdraw.aapolygon(self.screen, [(int(pt[0]), int(pt[1])) for pt in points], self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(pt[0]), int(pt[1])) for pt in points], self.COLOR_ENEMY)

            # Health bar
            if enemy['health'] < self.ENEMY_MAX_HEALTH:
                bar_w = 20
                bar_h = 4
                bar_x = enemy['pos'][0] - bar_w / 2
                bar_y = enemy['pos'][1] - self.ENEMY_SIZE - 8
                health_pct = enemy['health'] / self.ENEMY_MAX_HEALTH
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, (int(bar_x), int(bar_y), bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, (int(bar_x), int(bar_y), int(bar_w * health_pct), bar_h))

    def _render_projectiles(self):
        for proj in self.projectiles:
            color = self.COLOR_PLAYER_PROJ if proj['owner'] == 'player' else self.COLOR_ENEMY_PROJ
            # Trail
            for i, p in enumerate(proj['trail']):
                alpha = int(255 * (i + 1) / len(proj['trail']) * 0.5)
                trail_color = (*color, alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), int(self.PROJ_SIZE * 0.8), trail_color)
            # Head
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), self.PROJ_SIZE, color)
            pygame.gfxdraw.aacircle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), self.PROJ_SIZE, color)

    def _render_effects(self):
        # Particles
        for p in self.particles:
            if p['radius'] > 0:
                color = (*p['color'], int(255 * p['life'] / p['max_life']))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)
        
        # Floating texts
        for ft in self.floating_texts:
            alpha = int(255 * (ft['life'] / ft['max_life']))
            text_surf = self.font_popup.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (int(ft['pos'][0]), int(ft['pos'][1])))

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        # Player Health
        health_text_surf = self.font_ui.render("HEALTH", True, self.COLOR_UI)
        self.screen.blit(health_text_surf, (self.WIDTH - health_text_surf.get_width() - 10 - 150 - 10, 10))
        bar_w, bar_h = 150, 20
        bar_x, bar_y = self.WIDTH - bar_w - 10, 10
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, (50,0,0), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, int(bar_w * health_pct), bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI, (bar_x, bar_y, bar_w, bar_h), 1)

        # Stage
        if self.stage_clear_timer > 0:
            if self.stage <= self.STAGES:
                text = f"STAGE {self.stage-1} CLEAR"
            else:
                text = "VICTORY!"
            stage_surf = self.font_main.render(text, True, self.COLOR_UI)
            pos = (self.WIDTH/2 - stage_surf.get_width()/2, self.HEIGHT/2 - stage_surf.get_height()/2)
            self.screen.blit(stage_surf, pos)
        elif self.game_over:
            text = "GAME OVER"
            stage_surf = self.font_main.render(text, True, self.COLOR_ENEMY)
            pos = (self.WIDTH/2 - stage_surf.get_width()/2, self.HEIGHT/2 - stage_surf.get_height()/2)
            self.screen.blit(stage_surf, pos)
        else:
            stage_surf = self.font_ui.render(f"STAGE: {self.stage}/{self.STAGES}", True, self.COLOR_UI)
            self.screen.blit(stage_surf, (self.WIDTH/2 - stage_surf.get_width()/2, self.HEIGHT - 30))

    def _create_explosion(self, pos, count, colors, radius_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': life,
                'max_life': life,
                'radius': self.np_random.uniform(1, 3) * radius_mult,
                'decay': self.np_random.uniform(0.1, 0.3),
                'color': self.np_random.choice(colors)
            })
    
    def _create_floating_text(self, text, pos, color):
        life = 30
        self.floating_texts.append({
            'text': text,
            'pos': pos.copy(),
            'color': color,
            'life': life,
            'max_life': life
        })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "player_health": self.player_health,
            "enemies_left": len(self.enemies)
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Neon Arena")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # No-op
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        # --- End Human Controls ---

        # For random agent testing, uncomment below:
        # action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()