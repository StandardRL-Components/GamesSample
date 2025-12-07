
# Generated: 2025-08-27T17:23:47.556082
# Source Brief: brief_01516.md
# Brief Index: 1516

        
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
        "Controls: Use ←→ to move, ↑ to jump, and SPACE to attack."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced side-scrolling arcade game. Jump and slash your way through hordes of monsters to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GROUND_LEVEL = self.HEIGHT - 50
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 5
        self.PLAYER_JUMP_STRENGTH = -15
        self.MAX_STEPS = 1500 # Increased from 1000 for more gameplay
        self.WIN_CONDITION = 5

        # Colors
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_BG_LAYER1 = (25, 20, 50)
        self.COLOR_BG_LAYER2 = (40, 30, 70)
        self.COLOR_GROUND = (60, 45, 65)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_MONSTER = (255, 80, 80)
        self.COLOR_ATTACK = (255, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEALTH = (100, 255, 100)
        self.COLOR_HEALTH_BG = (100, 100, 100)

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.on_ground = None
        self.player_facing_right = None
        self.attack_timer = None
        self.attack_cooldown = None
        self.invulnerability_timer = None

        self.monsters = []
        self.player_attacks = []
        self.particles = []

        self.camera_x = 0
        self.monster_spawn_rate = 0.01
        self.monsters_defeated = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        
        # This check is not part of the standard API but is useful for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH // 4, self.GROUND_LEVEL)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 100
        self.on_ground = True
        self.player_facing_right = True
        self.attack_timer = 0
        self.attack_cooldown = 0
        self.invulnerability_timer = 0

        # Game state
        self.monsters = []
        self.player_attacks = []
        self.particles = []
        self.camera_x = 0
        self.monster_spawn_rate = 0.01
        self.monsters_defeated = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Spawn initial monster
        self._spawn_monster()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Unpack action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # Update timers
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        if self.invulnerability_timer > 0: self.invulnerability_timer -= 1
        if self.attack_timer > 0: self.attack_timer -= 1

        # Handle player input
        self._handle_input(movement, space_held)

        # Update game logic
        self._update_player()
        self._update_monsters()
        self._update_particles_and_attacks()
        
        # Handle collisions and calculate rewards
        reward += self._handle_collisions()

        # Handle monster spawning
        self._handle_spawning()
        
        # Small negative reward for inaction
        if not space_held:
            reward -= 0.02
        
        # Update camera
        target_camera_x = self.player_pos.x - self.WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

        self.steps += 1
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Horizontal Movement
        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_SPEED
            self.player_facing_right = False
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_SPEED
            self.player_facing_right = True
        else:
            self.player_vel.x = 0

        # Jumping
        if movement == 1 and self.on_ground: # Up
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            # SFX: Jump sound
            self._create_particles(self.player_pos.x, self.GROUND_LEVEL + 5, 5, self.COLOR_GROUND, count=5)

        # Attacking
        if space_held and self.attack_cooldown == 0:
            self.attack_cooldown = 20 # 20 frames cooldown
            self.attack_timer = 8 # Attack lasts 8 frames
            attack_x_offset = 40 if self.player_facing_right else -60
            attack_rect = pygame.Rect(self.player_pos.x + attack_x_offset, self.player_pos.y - 40, 60, 50)
            self.player_attacks.append(attack_rect)
            # SFX: Sword slash

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel
        
        # Ground collision
        if self.player_pos.y >= self.GROUND_LEVEL:
            if not self.on_ground: # Landing
                self._create_particles(self.player_pos.x, self.GROUND_LEVEL + 5, 5, self.COLOR_GROUND, count=8)
                # SFX: Land sound
            self.player_pos.y = self.GROUND_LEVEL
            self.player_vel.y = 0
            self.on_ground = True

        # Keep player within horizontal bounds of the "playable area"
        self.player_pos.x = max(self.camera_x + 20, self.player_pos.x)

    def _update_monsters(self):
        for monster in self.monsters:
            # Simple AI: move towards player
            if monster['pos'].x > self.player_pos.x:
                monster['pos'].x -= monster['speed']
            else:
                monster['pos'].x += monster['speed']

            # Attack logic
            monster['attack_cooldown'] = max(0, monster['attack_cooldown'] - 1)
            if abs(monster['pos'].x - self.player_pos.x) < 50 and monster['attack_cooldown'] == 0:
                monster['attack_cooldown'] = 90 # Slower attack rate for monsters
                self._create_particles(monster['pos'].x, monster['pos'].y - 20, 10, self.COLOR_MONSTER, count=5, life=10)
                # SFX: Monster preparing attack
                monster['is_attacking'] = 20 # Attack telegraph for 20 frames
            
            if monster['is_attacking'] > 0:
                monster['is_attacking'] -= 1

    def _update_particles_and_attacks(self):
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Player attacks expire with the attack_timer
        if self.attack_timer <= 0:
            self.player_attacks.clear()

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - 15, self.player_pos.y - 40, 30, 40)

        # Player attacks vs monsters
        for attack_rect in self.player_attacks:
            for monster in self.monsters[:]:
                monster_rect = pygame.Rect(monster['pos'].x - 15, monster['pos'].y - 40, 30, 40)
                if attack_rect.colliderect(monster_rect) and not monster.get('hit_this_swing', False):
                    monster['hit_this_swing'] = True
                    monster['health'] -= 10
                    reward += 0.1 # Reward for hitting
                    # SFX: Hit sound
                    self._create_particles(monster['pos'].x, monster['pos'].y - 20, 15, self.COLOR_ATTACK, count=10)
                    
                    if monster['health'] <= 0:
                        self._create_particles(monster['pos'].x, monster['pos'].y - 20, 20, self.COLOR_MONSTER, count=30)
                        self.monsters.remove(monster)
                        self.monsters_defeated += 1
                        self.score += 100
                        reward += 10 # Reward for kill
                        # SFX: Monster death sound
                    
        # Reset hit flag after swing is over
        if not self.player_attacks:
             for monster in self.monsters:
                 monster['hit_this_swing'] = False

        # Monster attacks vs player
        for monster in self.monsters:
            if monster['is_attacking'] == 1 and self.invulnerability_timer == 0: # Attack resolves on last frame
                monster_attack_rect = pygame.Rect(monster['pos'].x - 25, monster['pos'].y - 50, 50, 50)
                if monster_attack_rect.colliderect(player_rect):
                    self.player_health = max(0, self.player_health - 10)
                    self.invulnerability_timer = 30 # 0.5s invulnerability
                    reward -= 10 # Penalty for getting hit
                    self.score = max(0, self.score - 50)
                    # SFX: Player hurt sound
                    self._create_particles(self.player_pos.x, self.player_pos.y - 20, 15, self.COLOR_PLAYER, count=15)

        return reward

    def _handle_spawning(self):
        self.monster_spawn_rate = min(0.1, self.monster_spawn_rate + 0.0001)
        if self.np_random.random() < self.monster_spawn_rate and len(self.monsters) < 5:
            self._spawn_monster()

    def _spawn_monster(self):
        spawn_x = self.camera_x + self.WIDTH + self.np_random.integers(50, 150)
        self.monsters.append({
            'pos': pygame.Vector2(spawn_x, self.GROUND_LEVEL),
            'health': 20,
            'speed': self.np_random.uniform(1.5, 2.5),
            'attack_cooldown': self.np_random.integers(30, 90),
            'is_attacking': 0
        })

    def _check_termination(self):
        reward = 0
        terminated = False
        if self.player_health <= 0:
            terminated = True
            reward = -100
        elif self.monsters_defeated >= self.WIN_CONDITION:
            terminated = True
            reward = 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated:
            self.game_over = True
        
        return terminated, reward

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
            "player_health": self.player_health,
            "monsters_defeated": self.monsters_defeated,
        }

    def _create_particles(self, x, y, max_speed, color, count, life=20):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(x, y),
                'vel': pygame.Vector2(self.np_random.uniform(-max_speed, max_speed), self.np_random.uniform(-max_speed, max_speed)),
                'life': self.np_random.integers(life // 2, life),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _render_text(self, text, font, color, pos):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _render_game(self):
        # Parallax Background
        for i in range(10):
            # Layer 1 (Mid)
            x1 = int(100 * i - self.camera_x * 0.5) % (self.WIDTH + 200) - 100
            pygame.draw.rect(self.screen, self.COLOR_BG_LAYER1, (x1, self.np_random.integers(150, 250), 50, 200))
            # Layer 2 (Front)
            x2 = int(150 * i - self.camera_x * 0.8) % (self.WIDTH + 300) - 150
            pygame.draw.rect(self.screen, self.COLOR_BG_LAYER2, (x2, self.np_random.integers(200, 300), 80, 200))
        
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_LEVEL, self.WIDTH, self.HEIGHT - self.GROUND_LEVEL))

        # Render monsters
        for monster in self.monsters:
            m_pos_screen = (int(monster['pos'].x - self.camera_x), int(monster['pos'].y))
            if -50 < m_pos_screen[0] < self.WIDTH + 50:
                # Body
                body_rect = pygame.Rect(m_pos_screen[0] - 15, m_pos_screen[1] - 40, 30, 40)
                pygame.draw.rect(self.screen, self.COLOR_MONSTER, body_rect, border_radius=5)
                # Telegraph attack
                if monster['is_attacking'] > 0:
                    alpha = int(255 * (1 - (monster['is_attacking'] / 20.0)))
                    s = pygame.Surface((50, 50), pygame.SRCALPHA)
                    pygame.draw.rect(s, (*self.COLOR_ATTACK, alpha), (0,0,50,50), border_radius=8)
                    self.screen.blit(s, (m_pos_screen[0] - 25, m_pos_screen[1] - 50))

                # Health bar
                health_pct = max(0, monster['health'] / 20.0)
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (m_pos_screen[0] - 20, m_pos_screen[1] - 55, 40, 5))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH, (m_pos_screen[0] - 20, m_pos_screen[1] - 55, int(40 * health_pct), 5))

        # Render Player
        p_pos_screen = (int(self.player_pos.x - self.camera_x), int(self.player_pos.y))
        player_rect = pygame.Rect(p_pos_screen[0] - 15, p_pos_screen[1] - 40, 30, 40)
        
        # Invulnerability flash
        if self.invulnerability_timer > 0 and self.steps % 4 < 2:
            pass # Don't render player
        else:
            # Glow
            glow_rect = player_rect.inflate(20, 20)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_PLAYER_GLOW, (0, 0, glow_rect.width, glow_rect.height), border_radius=15)
            self.screen.blit(s, glow_rect.topleft)
            # Body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)

        # Render player attack
        if self.attack_timer > 0:
            direction = 1 if self.player_facing_right else -1
            lunge_offset = int(10 * (1 - (self.attack_timer / 8.0)) * direction)
            attack_x_offset = 20 if self.player_facing_right else -50
            attack_rect_screen = pygame.Rect(p_pos_screen[0] + attack_x_offset + lunge_offset, p_pos_screen[1] - 35, 40, 40)
            
            # Create a surface for the semi-transparent attack
            s = pygame.Surface(attack_rect_screen.size, pygame.SRCALPHA)
            alpha = int(200 * (self.attack_timer / 8.0))
            pygame.draw.rect(s, (*self.COLOR_ATTACK, alpha), (0,0, *attack_rect_screen.size), border_radius=10)
            self.screen.blit(s, attack_rect_screen.topleft)

        # Render particles
        for p in self.particles:
            p_pos_screen = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            alpha = max(0, int(255 * p['life'] / 20.0))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, p_pos_screen[0], p_pos_screen[1], int(p['radius']), color)
            pygame.gfxdraw.aacircle(self.screen, p_pos_screen[0], p_pos_screen[1], int(p['radius']), color)

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {self.score}", self.font_small, self.COLOR_TEXT, (10, 10))
        # Monsters defeated
        self._render_text(f"DEFEATED: {self.monsters_defeated}/{self.WIN_CONDITION}", self.font_small, self.COLOR_TEXT, (10, 30))

        # Health Bar
        health_pct = max(0, self.player_health / 100.0)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (self.WIDTH - bar_width - 10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (self.WIDTH - bar_width - 10, 10, int(bar_width * health_pct), 20))

        # Game Over / Win Message
        if self.game_over:
            message = "YOU WON!" if self.monsters_defeated >= self.WIN_CONDITION else "GAME OVER"
            color = self.COLOR_PLAYER if self.monsters_defeated >= self.WIN_CONDITION else self.COLOR_MONSTER
            text_surface = self.font_large.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    pygame.display.set_caption("Arcade Monster Slayer")
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Player controls
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()