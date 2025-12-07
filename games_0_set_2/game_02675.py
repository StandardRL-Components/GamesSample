
# Generated: 2025-08-28T05:35:01.166445
# Source Brief: brief_02675.md
# Brief Index: 2675

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→ to move, ↑ to jump. Press Space to fire your weapon."
    )

    game_description = (
        "Control a combat robot in a side-scrolling world. "
        "Destroy enemy drones, dodge their fire, and reach the exit to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.LEVEL_WIDTH = self.WIDTH * 4
        self.GROUND_Y = self.HEIGHT - 50
        self.FPS = 30
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_BG_LAYER1 = (25, 25, 40)
        self.COLOR_BG_LAYER2 = (40, 40, 60)
        self.COLOR_GROUND = (60, 50, 40)
        self.COLOR_PLAYER = (50, 200, 100)
        self.COLOR_PLAYER_DMG = (255, 100, 100)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_PLAYER_PROJ = (100, 200, 255)
        self.COLOR_ENEMY_PROJ = (255, 150, 50)
        self.COLOR_EXIT = (255, 220, 0)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HEALTH_BAR = (0, 200, 0)

        # Physics & Gameplay
        self.GRAVITY = 0.6
        self.PLAYER_SPEED = 4
        self.JUMP_STRENGTH = -13
        self.PLAYER_SHOOT_COOLDOWN = 10 # frames
        self.ENEMY_SHOOT_INTERVAL = 90 # frames (3 seconds)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.player = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.exit_rect = None
        self.steps = 0
        self.score = 0
        self.camera_x = 0
        self.target_camera_x = 0
        self.game_over = False
        self.game_won = False
        self.enemy_projectile_speed = 0.0
        
        self.reset()
        
        # Run validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.camera_x = 0
        self.target_camera_x = 0

        self.player = {
            'rect': pygame.Rect(100, self.GROUND_Y - 40, 28, 40),
            'vel': pygame.Vector2(0, 0),
            'health': 3,
            'max_health': 3,
            'on_ground': True,
            'facing_dir': 1,
            'shoot_cooldown': 0,
            'damage_timer': 0
        }

        self.enemies = []
        for _ in range(5):
            self._spawn_enemy()
            
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        self.exit_rect = pygame.Rect(self.LEVEL_WIDTH - 80, self.GROUND_Y - 80, 40, 80)
        self.enemy_projectile_speed = 3.0

        return self._get_observation(), self._get_info()
    
    def _spawn_enemy(self, respawn=False):
        min_x = self.WIDTH if respawn else 200
        x = self.np_random.integers(min_x, self.LEVEL_WIDTH - 100)
        
        # Avoid spawning on top of the player
        while self.player and abs(x - self.player['rect'].centerx) < self.WIDTH / 2:
            x = self.np_random.integers(min_x, self.LEVEL_WIDTH - 100)

        enemy = {
            'rect': pygame.Rect(x, self.GROUND_Y - 35, 35, 35),
            'vel_x': self.np_random.choice([-1.0, 1.0]),
            'patrol_center': x,
            'patrol_range': self.np_random.integers(50, 150),
            'shoot_timer': self.np_random.integers(0, self.ENEMY_SHOOT_INTERVAL)
        }
        self.enemies.append(enemy)

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        event_reward = 0
        
        if not self.game_over:
            # --- Handle Input & Player Logic ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            event_reward += self._handle_player(movement, space_held)

            # --- Update Game World ---
            self._handle_enemies()
            self._handle_projectiles()
            event_reward += self._handle_collisions()

        # --- Update Visuals & State ---
        self._update_particles()
        self._update_camera()
        self.steps += 1
        
        # --- Calculate Reward ---
        reward = event_reward
        reward -= 0.02  # Time penalty
        if not self.game_over and action[0] == 4: # Move right
             reward += 0.1
        self.score += event_reward # Score only reflects game events

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.player['rect'].colliderect(self.exit_rect):
                reward += 100
                self.score += 100
                self.game_won = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player(self, movement, space_held):
        # Horizontal Movement
        if movement == 3:  # Left
            self.player['vel'].x = -self.PLAYER_SPEED
            self.player['facing_dir'] = -1
        elif movement == 4:  # Right
            self.player['vel'].x = self.PLAYER_SPEED
            self.player['facing_dir'] = 1
        else:
            self.player['vel'].x = 0

        # Jumping
        if movement == 1 and self.player['on_ground']:
            self.player['vel'].y = self.JUMP_STRENGTH
            self.player['on_ground'] = False
            # Sound: Jump
            # Particle effect for jumping
            for _ in range(5):
                self.particles.append(self._create_particle(self.player['rect'].midbottom, (100, 90, 80), vel_y_range=(-2, -0.5)))

        # Shooting
        if self.player['shoot_cooldown'] > 0:
            self.player['shoot_cooldown'] -= 1
        if space_held and self.player['shoot_cooldown'] == 0:
            self.player['shoot_cooldown'] = self.PLAYER_SHOOT_COOLDOWN
            proj_start_x = self.player['rect'].right if self.player['facing_dir'] == 1 else self.player['rect'].left
            proj_start_y = self.player['rect'].centery - 5
            self.player_projectiles.append(pygame.Rect(proj_start_x, proj_start_y, 10, 4))
            # Sound: Shoot
            # Muzzle flash
            flash_pos = (proj_start_x + 5 * self.player['facing_dir'], proj_start_y + 2)
            for _ in range(8):
                self.particles.append(self._create_particle(flash_pos, self.COLOR_ENEMY_PROJ, vel_x_base=2 * self.player['facing_dir'], lifespan=5))

        # Physics
        self.player['vel'].y += self.GRAVITY
        self.player['rect'].x += int(self.player['vel'].x)
        self.player['rect'].y += int(self.player['vel'].y)
        
        # Ground Collision
        if self.player['rect'].bottom > self.GROUND_Y:
            if not self.player['on_ground']: # Landing
                # Sound: Land
                for _ in range(8):
                    self.particles.append(self._create_particle(self.player['rect'].midbottom, (100, 90, 80), vel_y_range=(-1.5, -0.2), vel_x_range=(-2, 2)))
            self.player['rect'].bottom = self.GROUND_Y
            self.player['vel'].y = 0
            self.player['on_ground'] = True

        # Level Boundaries
        self.player['rect'].left = max(0, self.player['rect'].left)
        self.player['rect'].right = min(self.LEVEL_WIDTH, self.player['rect'].right)
        
        if self.player['damage_timer'] > 0:
            self.player['damage_timer'] -= 1
        
        return 0 # No immediate reward from player actions

    def _handle_enemies(self):
        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.enemy_projectile_speed += 0.05

        for enemy in self.enemies:
            # Patrol
            enemy['rect'].x += enemy['vel_x']
            if enemy['rect'].centerx > enemy['patrol_center'] + enemy['patrol_range'] or \
               enemy['rect'].centerx < enemy['patrol_center'] - enemy['patrol_range']:
                enemy['vel_x'] *= -1

            # Shoot
            enemy['shoot_timer'] += 1
            if enemy['shoot_timer'] > self.ENEMY_SHOOT_INTERVAL:
                enemy['shoot_timer'] = 0
                # Only shoot if player is somewhat nearby
                if abs(enemy['rect'].centerx - self.player['rect'].centerx) < self.WIDTH * 0.75:
                    dir_to_player = 1 if self.player['rect'].centerx > enemy['rect'].centerx else -1
                    proj_start_pos = enemy['rect'].center
                    self.enemy_projectiles.append({
                        'rect': pygame.Rect(proj_start_pos[0], proj_start_pos[1], 8, 8),
                        'vel_x': dir_to_player * self.enemy_projectile_speed
                    })
                    # Sound: Enemy Shoot

    def _handle_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj.x += 8 * self.player['facing_dir']
            if proj.right < 0 or proj.left > self.LEVEL_WIDTH:
                self.player_projectiles.remove(proj)
        # Enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj['rect'].x += proj['vel_x']
            if proj['rect'].right < 0 or proj['rect'].left > self.LEVEL_WIDTH:
                self.enemy_projectiles.remove(proj)

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj.colliderect(enemy['rect']):
                    self.player_projectiles.remove(proj)
                    self.enemies.remove(enemy)
                    # Sound: Explosion
                    for _ in range(20):
                        self.particles.append(self._create_particle(enemy['rect'].center, self.COLOR_ENEMY, lifespan=25))
                    reward += 10
                    self._spawn_enemy(respawn=True)
                    assert len(self.enemies) == 5
                    break
        
        # Enemy projectiles vs player
        if self.player['damage_timer'] == 0:
            for proj in self.enemy_projectiles[:]:
                if self.player['rect'].colliderect(proj['rect']):
                    self.enemy_projectiles.remove(proj)
                    self.player['health'] -= 1
                    assert self.player['health'] <= self.player['max_health']
                    self.player['damage_timer'] = 30 # 1s invulnerability
                    # Sound: Player Hit
                    for _ in range(15):
                        self.particles.append(self._create_particle(self.player['rect'].center, (200, 200, 255), lifespan=20))
                    reward -= 5
                    break
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _update_camera(self):
        self.target_camera_x = self.player['rect'].centerx - self.WIDTH / 2
        self.target_camera_x = max(0, min(self.LEVEL_WIDTH - self.WIDTH, self.target_camera_x))
        # Smooth camera movement
        self.camera_x += (self.target_camera_x - self.camera_x) * 0.1

    def _check_termination(self):
        if self.player['health'] <= 0:
            return True
        if self.player['rect'].colliderect(self.exit_rect):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Parallax Layer 1
        scroll_x1 = -(self.camera_x * 0.2)
        for i in range(math.ceil(self.WIDTH / 100) + 1):
            for j in range(math.ceil(self.HEIGHT / 100) + 1):
                if (i + j) % 2 == 0:
                    pygame.draw.rect(self.screen, self.COLOR_BG_LAYER1, 
                                     (i * 100 + scroll_x1 % 100 - 100, j * 100, 50, 50))

        # Parallax Layer 2 (Vertical Girders)
        scroll_x2 = -(self.camera_x * 0.5)
        for i in range(math.ceil(self.LEVEL_WIDTH / 200)):
            x = i * 200 - self.camera_x
            if -40 < x < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_BG_LAYER2, (x, 0, 40, self.HEIGHT))
                pygame.draw.rect(self.screen, self.COLOR_BG, (x+5, 0, 30, self.HEIGHT))

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

    def _render_game(self):
        # Helper to convert world to screen coordinates
        def to_screen(rect):
            return rect.move(-int(self.camera_x), 0)

        # Exit
        pygame.draw.rect(self.screen, self.COLOR_EXIT, to_screen(self.exit_rect))
        pygame.draw.rect(self.screen, self.COLOR_BG, to_screen(self.exit_rect.inflate(-10, -10)))
        
        # Enemies
        for enemy in self.enemies:
            e_rect_s = to_screen(enemy['rect'])
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, e_rect_s)
            pygame.draw.rect(self.screen, (255,150,150), e_rect_s.inflate(-10, -10)) # "Eye"

        # Player Projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, to_screen(proj))

        # Enemy Projectiles
        for proj in self.enemy_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_ENEMY_PROJ, to_screen(proj['rect']).center, 5)

        # Player
        p_rect_s = to_screen(self.player['rect'])
        player_color = self.COLOR_PLAYER if self.player['damage_timer'] % 10 < 5 else self.COLOR_PLAYER_DMG
        # Body with squash/stretch
        squash = 1.0
        if not self.player['on_ground']:
            squash = 1.0 - max(-1, min(1, self.player['vel'].y / 20)) # Stretch on fall/rise
        body_h = self.player['rect'].height * squash
        body_w = self.player['rect'].width / squash
        body_rect = pygame.Rect(0, 0, body_w, body_h)
        body_rect.midbottom = p_rect_s.midbottom
        pygame.draw.rect(self.screen, player_color, body_rect, border_radius=4)
        # Gun
        gun_y = body_rect.centery - 5
        gun_w = 12
        gun_h = 8
        gun_x = body_rect.centerx if self.player['facing_dir'] == 1 else body_rect.centerx - gun_w
        pygame.draw.rect(self.screen, (100,100,120), (gun_x, gun_y, gun_w, gun_h))
        
        # Particles
        for p in self.particles:
            pos = [p['pos'][0] - self.camera_x, p['pos'][1]]
            size = max(1, p['lifespan'] / 5)
            pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Health Bar
        bar_w, bar_h = 150, 20
        health_pct = max(0, self.player['health'] / self.player['max_health'])
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_w * health_pct, bar_h))
        
        # Game Over / You Win
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_EXIT if self.game_won else self.COLOR_ENEMY
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, self.COLOR_BG, text_rect.inflate(20, 20))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player['health'],
            "player_pos": (self.player['rect'].x, self.player['rect'].y)
        }

    def _create_particle(self, pos, color, vel_x_base=0, vel_x_range=(-3, 3), vel_y_range=(-3, 3), lifespan=15):
        return {
            'pos': list(pos),
            'vel': [vel_x_base + self.np_random.uniform(*vel_x_range), self.np_random.uniform(*vel_y_range)],
            'color': color,
            'lifespan': self.np_random.integers(lifespan * 0.8, lifespan * 1.2)
        }
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Robo-Action Platformer")
    
    running = True
    total_reward = 0
    
    # --- Instructions ---
    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")
    
    while running:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            print("Press 'R' to restart.")

    env.close()