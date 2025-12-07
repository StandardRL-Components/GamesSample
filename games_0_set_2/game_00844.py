
# Generated: 2025-08-27T14:57:13.982027
# Source Brief: brief_00844.md
# Brief Index: 844

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    A side-scrolling zombie shooter environment for Gymnasium.

    **Objective:** Survive and eliminate all zombies.
    **Actions:** Move left/right and shoot.
    **Rewards:**
        - +100 for winning (killing all zombies).
        - -100 for losing (health reaches 0).
        - +1 for each zombie killed.
        - -1 for each time the player is hit.
        - -0.01 per step to encourage efficiency.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move. Press space to fire your weapon."
    )

    # Short, user-facing description of the game
    game_description = (
        "Blast hordes of procedurally generated zombies in a side-scrolling arcade shooter."
    )

    # Frames auto-advance at 30fps for real-time gameplay
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GROUND_Y = HEIGHT - 50
    FPS = 30
    MAX_STEPS = 1000
    TOTAL_ZOMBIES = 20

    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_GROUND = (40, 30, 30)
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_EYE = (255, 255, 255)
    COLOR_ZOMBIE = (200, 50, 50)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_OVERLAY = (0, 0, 0, 180)

    # Player settings
    PLAYER_WIDTH, PLAYER_HEIGHT = 20, 40
    PLAYER_SPEED = 6
    PLAYER_MAX_HEALTH = 5
    PLAYER_HIT_INVULNERABILITY = 30  # frames

    # Zombie settings
    ZOMBIE_WIDTH, ZOMBIE_HEIGHT = 20, 40
    ZOMBIE_BASE_SPEED = 1.0
    ZOMBIE_SPEED_INCREASE = 0.02

    # Projectile settings
    PROJECTILE_SPEED = 10
    PROJECTILE_COOLDOWN = 6  # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 60, bold=True)

        # Initialize state variables
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_facing_dir = 1
        self.player_hit_cooldown = 0
        self.player_bob = 0.0

        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.zombies_killed = 0
        self.zombie_speed = self.ZOMBIE_BASE_SPEED
        self.shoot_cooldown = 0
        self.game_over = False
        self.win = False
        self.bg_layers = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = [self.WIDTH / 2, self.GROUND_Y]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing_dir = 1
        self.player_hit_cooldown = 0
        self.player_bob = 0.0

        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.shoot_cooldown = 0
        self.zombies_killed = 0
        self.zombie_speed = self.ZOMBIE_BASE_SPEED
        
        self._spawn_zombies(self.TOTAL_ZOMBIES)
        self._generate_background()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Time penalty
        
        # --- Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held)

        # --- Update Game State ---
        self._update_player()
        self._update_projectiles()
        self._update_zombies()
        self._update_particles()
        
        # --- Handle Collisions & Events ---
        reward += self._handle_collisions()

        # --- Check Termination ---
        self.steps += 1
        terminated = False
        
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 100
        elif self.zombies_killed == self.TOTAL_ZOMBIES:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win = False

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
            self.player_facing_dir = -1
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
            self.player_facing_dir = 1
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_WIDTH/2, self.WIDTH - self.PLAYER_WIDTH/2)

        # Shooting
        if space_held and self.shoot_cooldown == 0:
            proj_start_pos = [self.player_pos[0] + self.player_facing_dir * (self.PLAYER_WIDTH/2 + 5), self.player_pos[1] - self.PLAYER_HEIGHT/2]
            self.projectiles.append({'pos': proj_start_pos, 'dir': self.player_facing_dir})
            self.shoot_cooldown = self.PROJECTILE_COOLDOWN
            # Muzzle flash
            self._create_particles(proj_start_pos, self.COLOR_PROJECTILE, 15, 1, 5, 0.9)
            # Placeholder: # Play shoot_sound

    def _update_player(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.player_hit_cooldown > 0:
            self.player_hit_cooldown -= 1
        self.player_bob = math.sin(self.steps * 0.2) * 2

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['pos'][0] += proj['dir'] * self.PROJECTILE_SPEED
            if not (0 < proj['pos'][0] < self.WIDTH):
                self.projectiles.remove(proj)

    def _update_zombies(self):
        self.zombie_speed = self.ZOMBIE_BASE_SPEED + (self.zombies_killed // 5) * self.ZOMBIE_SPEED_INCREASE
        for z in self.zombies:
            direction = 1 if self.player_pos[0] > z['pos'][0] else -1
            z['pos'][0] += direction * self.zombie_speed
            z['anim_offset'] += 0.2
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            p['radius'] *= p['decay']
            if p['life'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_WIDTH/2, self.player_pos[1] - self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        # Projectiles vs Zombies
        for proj in self.projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'][0] - 2, proj['pos'][1] - 2, 4, 4)
            for z in self.zombies[:]:
                zombie_rect = pygame.Rect(z['pos'][0] - self.ZOMBIE_WIDTH/2, z['pos'][1] - self.ZOMBIE_HEIGHT, self.ZOMBIE_WIDTH, self.ZOMBIE_HEIGHT)
                if proj_rect.colliderect(zombie_rect):
                    self.zombies.remove(z)
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    self.zombies_killed += 1
                    self.score += 10
                    reward += 1
                    self._create_particles(z['pos'], self.COLOR_ZOMBIE, 20, 0.5, 3, 0.95)
                    # Placeholder: # Play zombie_die_sound
                    break

        # Zombies vs Player
        if self.player_hit_cooldown == 0:
            for z in self.zombies:
                zombie_rect = pygame.Rect(z['pos'][0] - self.ZOMBIE_WIDTH/2, z['pos'][1] - self.ZOMBIE_HEIGHT, self.ZOMBIE_WIDTH, self.ZOMBIE_HEIGHT)
                if player_rect.colliderect(zombie_rect):
                    self.player_health -= 1
                    self.player_hit_cooldown = self.PLAYER_HIT_INVULNERABILITY
                    reward -= 1
                    self._create_particles(self.player_pos, self.COLOR_PLAYER, 10, 0.5, 2, 0.9)
                    # Placeholder: # Play player_hit_sound
                    break
        return reward

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
            "health": self.player_health,
            "zombies_remaining": self.TOTAL_ZOMBIES - self.zombies_killed,
        }

    def _spawn_zombies(self, count):
        for i in range(count):
            side = random.choice([-1, 1])
            x_pos = self.WIDTH/2 + side * random.uniform(self.WIDTH/2 + 50, self.WIDTH)
            self.zombies.append({
                'pos': [x_pos, self.GROUND_Y],
                'anim_offset': random.uniform(0, math.pi * 2)
            })

    def _generate_background(self):
        self.bg_layers = []
        for i in range(3):
            layer = []
            for _ in range(20):
                w = random.randint(30, 80)
                h = random.randint(50, 200 - i * 40)
                x = random.randint(-self.WIDTH, self.WIDTH * 2)
                layer.append(pygame.Rect(x, self.GROUND_Y - h, w, h))
            self.bg_layers.append({'rects': layer, 'color': (20 + i*5, 20 + i*5, 35 + i*10)})

    def _create_particles(self, pos, color, count, min_speed, max_speed, decay):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': random.randint(10, 20),
                'color': color,
                'radius': random.uniform(2, 5),
                'decay': decay
            })

    def _render_background(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        for layer in self.bg_layers:
            for rect in layer['rects']:
                pygame.draw.rect(self.screen, layer['color'], rect)

    def _render_game(self):
        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, int(p['radius'])), p['color'])
        
        # Zombies
        for z in self.zombies:
            bob = math.sin(z['anim_offset']) * 2
            z_rect = pygame.Rect(
                z['pos'][0] - self.ZOMBIE_WIDTH / 2,
                z['pos'][1] - self.ZOMBIE_HEIGHT + bob,
                self.ZOMBIE_WIDTH,
                self.ZOMBIE_HEIGHT
            )
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z_rect)

        # Player
        if self.player_hit_cooldown > 0 and (self.steps // 2) % 2 == 0:
            pass # Flicker on hit
        else:
            p_rect = pygame.Rect(
                self.player_pos[0] - self.PLAYER_WIDTH / 2,
                self.player_pos[1] - self.PLAYER_HEIGHT + self.player_bob,
                self.PLAYER_WIDTH,
                self.PLAYER_HEIGHT
            )
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect)
            # Eye
            eye_x = self.player_pos[0] + self.player_facing_dir * 5
            eye_y = self.player_pos[1] - self.PLAYER_HEIGHT * 0.7 + self.player_bob
            pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, (int(eye_x), int(eye_y)), 2)
        
        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 200, 20))
        if health_ratio > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, 200 * health_ratio, 20))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Zombies remaining
        zombie_text = self.font_small.render(f"ZOMBIES: {self.TOTAL_ZOMBIES - self.zombies_killed}", True, self.COLOR_TEXT)
        self.screen.blit(zombie_text, (self.WIDTH - zombie_text.get_width() - 10, 30))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Must be set for pygame to run headlessly
    
    env = GameEnv(render_mode="rgb_array")
    
    # To run with manual controls:
    # 1. pip install pygame
    # 2. Comment out the os.environ["SDL_VIDEODRIVER"] line above.
    # 3. Uncomment the code below.
    
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Zombie Shooter")
    # clock = pygame.time.Clock()

    # obs, info = env.reset()
    # done = False
    # while not done:
    #     movement, space, shift = 0, 0, 0
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True

    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]:
    #         movement = 3
    #     elif keys[pygame.K_RIGHT]:
    #         movement = 4
        
    #     if keys[pygame.K_SPACE]:
    #         space = 1
        
    #     action = [movement, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     # Render the observation to the display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()

    #     clock.tick(env.FPS)
    
    # print(f"Game Over! Final Score: {info['score']}")
    # env.close()