
# Generated: 2025-08-28T02:49:27.678170
# Source Brief: brief_01827.md
# Brief Index: 1827

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to fire your weapon. Survive the horde and reach the exit!"
    )
    game_description = (
        "Escape hordes of procedurally generated zombies in a side-scrolling survival shooter. Reach the exit to win."
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    # Screen and Level
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    LEVEL_WIDTH = SCREEN_WIDTH * 5
    GROUND_Y = 350
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_BRICK_DARK = (40, 45, 50)
    COLOR_BRICK_LIGHT = (50, 55, 60)
    COLOR_GROUND = (60, 65, 70)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GUN = (200, 200, 200)
    COLOR_ZOMBIE = (80, 140, 60)
    COLOR_ZOMBIE_DAMAGED = (180, 80, 60)
    COLOR_BULLET = (255, 255, 255)
    COLOR_MUZZLE_FLASH = (255, 220, 100)
    COLOR_BLOOD = (200, 40, 40)
    COLOR_EXIT = (100, 255, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)

    # Physics and Gameplay
    GRAVITY = 0.6
    PLAYER_SPEED = 5
    PLAYER_JUMP_STRENGTH = -12
    PLAYER_MAX_HEALTH = 50
    BULLET_SPEED = 15
    SHOOT_COOLDOWN = 6  # frames
    ZOMBIE_BASE_SPEED = 1.0
    ZOMBIE_DAMAGE = 10
    ZOMBIE_COUNT = 10
    ZOMBIE_SPAWN_START_X = SCREEN_WIDTH + 100
    ZOMBIE_SPAWN_END_X = LEVEL_WIDTH - 200

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_health = 0
        self.player_on_ground = False
        self.player_facing_right = True
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.camera_x = 0
        self.shoot_cooldown_timer = 0
        self.zombie_speed_modifier = 0

        self.exit_rect = pygame.Rect(self.LEVEL_WIDTH - 100, self.GROUND_Y - 100, 50, 100)

        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = pygame.math.Vector2(100, self.GROUND_Y)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_on_ground = True
        self.player_facing_right = True

        self.zombies = []
        self.bullets = []
        self.particles = []
        self._spawn_zombies()
        
        self.camera_x = 0
        self.shoot_cooldown_timer = 0
        self.zombie_speed_modifier = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage speed

        if not self.game_over:
            # --- Action Handling ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            self._handle_input(movement, space_held)

            # --- Game Logic Update ---
            self._update_player()
            self._update_zombies()
            self._update_bullets()
            self._update_particles()
            
            # --- Collision Detection ---
            collision_rewards = self._handle_collisions()
            reward += collision_rewards

            # --- Difficulty Scaling ---
            if self.steps % 500 == 0 and self.steps > 0:
                self.zombie_speed_modifier += 0.01

        # --- Termination Check ---
        self.terminated = self._check_termination()
        if self.terminated and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 100
            else: # Player died
                reward += -100
        
        # --- Finalization ---
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
            self.player_facing_right = False
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
            self.player_facing_right = True
        else:
            self.player_vel.x = 0
        
        # Jumping
        if movement == 1 and self.player_on_ground:
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.player_on_ground = False
            # Sound: Player Jump
        
        # Shooting
        if self.shoot_cooldown_timer > 0:
            self.shoot_cooldown_timer -= 1

        if space_held and self.shoot_cooldown_timer == 0:
            self._fire_bullet()
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN

    def _fire_bullet(self):
        # Sound: Gunshot
        direction = 1 if self.player_facing_right else -1
        gun_offset_x = 25 if self.player_facing_right else -25
        gun_offset_y = -25
        start_pos = self.player_pos + pygame.math.Vector2(gun_offset_x, gun_offset_y)
        self.bullets.append({'pos': start_pos, 'dir': direction})
        
        # Muzzle flash
        for _ in range(10):
            vel = pygame.math.Vector2(random.uniform(2, 5) * direction, random.uniform(-2, 2))
            self.particles.append({
                'pos': start_pos.copy(), 'vel': vel, 'life': 5, 
                'color': self.COLOR_MUZZLE_FLASH, 'radius': random.randint(2, 4)
            })

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel
        
        # Ground collision
        if self.player_pos.y > self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y
            self.player_vel.y = 0
            self.player_on_ground = True
        
        # Level boundaries
        self.player_pos.x = max(15, min(self.player_pos.x, self.LEVEL_WIDTH - 15))

    def _spawn_zombies(self):
        for _ in range(self.ZOMBIE_COUNT):
            pos_x = random.randint(self.ZOMBIE_SPAWN_START_X, self.ZOMBIE_SPAWN_END_X)
            self.zombies.append({
                'pos': pygame.math.Vector2(pos_x, self.GROUND_Y),
                'vel': pygame.math.Vector2(0, 0),
                'on_ground': True,
                'health': 1
            })

    def _update_zombies(self):
        current_speed = self.ZOMBIE_BASE_SPEED + self.zombie_speed_modifier
        for zombie in self.zombies:
            # AI: Move towards player
            direction = self.player_pos.x - zombie['pos'].x
            if abs(direction) > 1: # Dead zone to prevent jittering
                zombie['vel'].x = math.copysign(current_speed, direction)
            else:
                zombie['vel'].x = 0

            # Physics
            zombie['vel'].y += self.GRAVITY
            zombie['pos'] += zombie['vel']

            # Ground collision
            if zombie['pos'].y > self.GROUND_Y:
                zombie['pos'].y = self.GROUND_Y
                zombie['vel'].y = 0
                zombie['on_ground'] = True

    def _update_bullets(self):
        self.bullets = [
            b for b in self.bullets 
            if 0 < b['pos'].x < self.LEVEL_WIDTH
        ]
        for bullet in self.bullets:
            bullet['pos'].x += self.BULLET_SPEED * bullet['dir']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['color'] == self.COLOR_BLOOD:
                p['vel'] *= 0.9 # Friction for blood

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - 15, self.player_pos.y - 40, 30, 40)

        # Player vs Zombies
        for zombie in self.zombies:
            zombie_rect = pygame.Rect(zombie['pos'].x - 15, zombie['pos'].y - 40, 30, 40)
            if player_rect.colliderect(zombie_rect):
                self.player_health -= self.ZOMBIE_DAMAGE
                # Sound: Player Hurt
                # Knockback
                knockback_dir = math.copysign(1, player_rect.centerx - zombie_rect.centerx)
                self.player_vel.x += knockback_dir * 5
                self.player_vel.y = -5 # Pop up
                self.player_health = max(0, self.player_health)
                break # Only one hit per frame

        # Bullets vs Zombies
        zombies_hit_indices = set()
        bullets_to_remove = []
        for i, bullet in enumerate(self.bullets):
            bullet_rect = pygame.Rect(bullet['pos'].x - 2, bullet['pos'].y - 2, 4, 4)
            for j, zombie in enumerate(self.zombies):
                if j in zombies_hit_indices: continue
                zombie_rect = pygame.Rect(zombie['pos'].x - 15, zombie['pos'].y - 40, 30, 40)
                if bullet_rect.colliderect(zombie_rect):
                    zombies_hit_indices.add(j)
                    bullets_to_remove.append(i)
                    break
        
        if zombies_hit_indices:
            # Sound: Zombie Hit/Die
            for index in sorted(list(zombies_hit_indices), reverse=True):
                # Create blood splatter
                pos = self.zombies[index]['pos']
                for _ in range(20):
                    angle = random.uniform(0, 2 * math.pi)
                    speed = random.uniform(1, 5)
                    vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
                    self.particles.append({
                        'pos': pos.copy() - (0, 20), 'vel': vel, 'life': random.randint(15, 30),
                        'color': self.COLOR_BLOOD, 'radius': random.randint(1, 3)
                    })
                del self.zombies[index]
                self.score += 10
                reward += 1.0

        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]
        
        # Player vs Exit
        if player_rect.colliderect(self.exit_rect):
            self.win = True

        return reward

    def _check_termination(self):
        return self.player_health <= 0 or self.win or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        # Update camera to follow player
        self.camera_x = self.player_pos.x - self.SCREEN_WIDTH / 2
        self.camera_x = max(0, min(self.camera_x, self.LEVEL_WIDTH - self.SCREEN_WIDTH))

        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_background()
        self._render_game_objects()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw brick pattern
        brick_w, brick_h = 80, 40
        start_x = -int(self.camera_x % brick_w) - brick_w
        start_y = 0
        
        for y in range(start_y, self.SCREEN_HEIGHT, brick_h):
            row_offset = (y // brick_h) % 2 * (brick_w // 2)
            for x in range(start_x, self.SCREEN_WIDTH, brick_w):
                brick_rect = pygame.Rect(x - row_offset, y, brick_w, brick_h)
                pygame.draw.rect(self.screen, self.COLOR_BRICK_LIGHT, brick_rect)
                pygame.draw.rect(self.screen, self.COLOR_BRICK_DARK, brick_rect, 1)

        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

    def _render_game_objects(self):
        # Exit
        exit_screen_rect = self.exit_rect.move(-self.camera_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_screen_rect)
        pygame.gfxdraw.rectangle(self.screen, exit_screen_rect, tuple(c*0.8 for c in self.COLOR_EXIT))

        # Zombies
        for z in self.zombies:
            x, y = int(z['pos'].x - self.camera_x), int(z['pos'].y)
            bob = math.sin(self.steps * 0.1 + z['pos'].x * 0.1) * 2
            z_rect = pygame.Rect(x - 15, y - 40 + bob, 30, 40)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z_rect)

        # Player
        px, py = int(self.player_pos.x - self.camera_x), int(self.player_pos.y)
        bob = 0
        if self.player_vel.x != 0 and self.player_on_ground:
            bob = math.sin(self.steps * 0.4) * 3
        player_rect = pygame.Rect(px - 15, py - 40 + bob, 30, 40)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Gun
        gun_w, gun_h = (20, 6)
        gun_x = px + 10 if self.player_facing_right else px - 10 - gun_w
        gun_y = py - 28 + bob
        gun_rect = pygame.Rect(gun_x, gun_y, gun_w, gun_h)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GUN, gun_rect)

        # Bullets
        for b in self.bullets:
            bx, by = int(b['pos'].x - self.camera_x), int(b['pos'].y)
            pygame.draw.circle(self.screen, self.COLOR_BULLET, (bx, by), 3)

        # Particles
        for p in self.particles:
            px, py = int(p['pos'].x - self.camera_x), int(p['pos'].y)
            alpha = max(0, min(255, int(255 * (p['life'] / 15.0))))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(p['radius']), color)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_ratio), 20))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU REACHED THE EXIT!" if self.win else "YOU DIED"
            color = self.COLOR_EXIT if self.win else self.COLOR_BLOOD
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,180), text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "zombies_remaining": len(self.zombies),
            "player_pos": (self.player_pos.x, self.player_pos.y),
        }

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Test for 100 steps with random actions
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"Episode finished after {i+1} steps.")
            break
    
    print(f"Test run complete. Final info: {info}")
    print(f"Total reward over 100 steps: {total_reward}")

    # To visualize the game, you would need to run it with a display driver
    # and render the frames from _get_observation() to the screen.
    # Example (requires a display):
    #
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # clock = pygame.time.Clock()
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #     
    #     # Simple keyboard mapping for human play
    #     keys = pygame.key.get_pressed()
    #     move = 0
    #     if keys[pygame.K_UP]: move = 1
    #     if keys[pygame.K_LEFT]: move = 3
    #     if keys[pygame.K_RIGHT]: move = 4
    #     space = 1 if keys[pygame.K_SPACE] else 0
    #     shift = 1 if keys[pygame.K_LSHIFT] else 0
    #     action = [move, space, shift]
    #     
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     
    #     # Render to screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #     
    #     if terminated:
    #         print("Game Over! Resetting in 3 seconds...")
    #         pygame.time.wait(3000)
    #         env.reset()
    #     
    #     clock.tick(30) # 30 FPS
    # env.close()