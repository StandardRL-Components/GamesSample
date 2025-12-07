
# Generated: 2025-08-27T19:51:51.915100
# Source Brief: brief_02280.md
# Brief Index: 2280

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space to fire. Press shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless zombie horde for 60 seconds in this top-down arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_TIME = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_GLOW = (255, 100, 100, 50)
    COLOR_ZOMBIE = (50, 200, 50)
    COLOR_ZOMBIE_GLOW = (100, 255, 100, 40)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR_BG = (70, 0, 0)
    COLOR_HEALTH_BAR_FG = (220, 0, 0)
    COLOR_RELOAD = (255, 165, 0)

    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 4.0
    PLAYER_MAX_HEALTH = 100
    PLAYER_MAX_AMMO = 30
    SHOOT_COOLDOWN = 5  # frames
    RELOAD_TIME = 30 # frames (1 second)

    # Zombie
    ZOMBIE_SIZE = 14
    ZOMBIE_SPEED = 1.0
    ZOMBIE_HEALTH = 10
    ZOMBIE_DAMAGE = 1
    ZOMBIE_SPAWN_INTERVAL = 60 # frames (2 seconds)
    DIFFICULTY_INTERVAL = 450 # frames (15 seconds)

    # Projectile
    PROJECTILE_SPEED = 10.0
    PROJECTILE_DAMAGE = 10
    PROJECTILE_WIDTH = 3
    PROJECTILE_HEIGHT = 10

    # Particles
    PARTICLE_LIFESPAN = 15
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        self.render_mode = render_mode
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.ammo = self.PLAYER_MAX_AMMO
        self.last_move_direction = np.array([0, -1], dtype=np.float32) # Start aiming up

        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.time_left = self.MAX_TIME
        self.zombie_spawn_timer = self.ZOMBIE_SPAWN_INTERVAL
        self.zombies_to_spawn = 1
        
        self.shoot_cooldown_timer = 0
        self.reload_timer = 0
        
        # Track previous action state for edge triggers
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        # --- Update Timers ---
        self.steps += 1
        self.time_left -= 1
        self.shoot_cooldown_timer = max(0, self.shoot_cooldown_timer - 1)
        if self.reload_timer > 0:
            self.reload_timer -= 1
            if self.reload_timer == 0:
                self.ammo = self.PLAYER_MAX_AMMO
                # sfx: reload_complete.wav

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_movement(movement)
        
        # Use edge trigger for shooting
        if space_held and not self.prev_space_held:
            shot_fired, shot_hit = self._handle_shooting()
            if shot_fired and not shot_hit: # This check is for missed shots
                # The reward for missed shots is handled when projectiles despawn
                pass
        
        # Use edge trigger for reloading
        if shift_held and not self.prev_shift_held:
            if self._handle_reloading():
                reward += 5 # Reward for smart reloading

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game Logic ---
        reward += self._update_projectiles() # Returns reward from kills/misses
        self._update_zombies()
        self._spawn_zombies()
        self._update_difficulty()
        self._update_particles()
        
        # --- Collisions ---
        self._handle_collisions()

        # --- Reward & Termination ---
        reward += 0.1  # Survival reward per frame

        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward -= 100
            # sfx: player_death.wav
        
        if self.time_left <= 0 and not self.game_over:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100
            self.score += 1000
            # sfx: level_win.wav

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        move_vector = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vector[1] -= 1 # Up
        if movement == 2: move_vector[1] += 1 # Down
        if movement == 3: move_vector[0] -= 1 # Left
        if movement == 4: move_vector[0] += 1 # Right

        if np.linalg.norm(move_vector) > 0:
            move_vector /= np.linalg.norm(move_vector)
            self.player_pos += move_vector * self.PLAYER_SPEED
            self.last_move_direction = move_vector

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _handle_shooting(self):
        if self.ammo > 0 and self.shoot_cooldown_timer == 0 and self.reload_timer == 0:
            self.ammo -= 1
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
            
            proj_start_pos = self.player_pos + self.last_move_direction * self.PLAYER_SIZE
            projectile = {
                'pos': proj_start_pos,
                'dir': self.last_move_direction,
                'hit': False
            }
            self.projectiles.append(projectile)
            # sfx: shoot.wav
            
            # Muzzle flash
            for _ in range(10):
                self.particles.append(self._create_particle(
                    proj_start_pos, 
                    color=(255, 220, 150), 
                    lifespan=5,
                    speed_mult=3.0,
                    base_vel=self.last_move_direction * 2
                ))
            return True, False
        return False, False

    def _handle_reloading(self):
        if self.reload_timer == 0 and self.ammo < self.PLAYER_MAX_AMMO:
            self.reload_timer = self.RELOAD_TIME
            # sfx: reload_start.wav
            return self.ammo < 10
        return False

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p['pos'] += p['dir'] * self.PROJECTILE_SPEED
            if not (0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT):
                if not p['hit']:
                    reward -= 0.2 # Penalty for missed shot
                self.projectiles.remove(p)
        return reward

    def _update_zombies(self):
        for z in self.zombies:
            direction = self.player_pos - z['pos']
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction /= dist
            z['pos'] += direction * self.ZOMBIE_SPEED
            if z['hit_timer'] > 0:
                z['hit_timer'] -= 1

    def _spawn_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            self.zombie_spawn_timer = self.ZOMBIE_SPAWN_INTERVAL
            for _ in range(self.zombies_to_spawn):
                edge = self.np_random.integers(4)
                if edge == 0: # Top
                    pos = [self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE]
                elif edge == 1: # Bottom
                    pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE]
                elif edge == 2: # Left
                    pos = [-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)]
                else: # Right
                    pos = [self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)]
                
                self.zombies.append({
                    'pos': np.array(pos, dtype=np.float32),
                    'health': self.ZOMBIE_HEALTH,
                    'hit_timer': 0
                })
            # sfx: zombie_spawn.wav

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.zombies_to_spawn += 1

    def _handle_collisions(self):
        # Projectiles vs Zombies
        reward = 0
        for p in self.projectiles[:]:
            proj_rect = pygame.Rect(p['pos'][0] - self.PROJECTILE_WIDTH/2, p['pos'][1] - self.PROJECTILE_HEIGHT/2, self.PROJECTILE_WIDTH, self.PROJECTILE_HEIGHT)
            
            for z in self.zombies[:]:
                zombie_rect = pygame.Rect(z['pos'][0] - self.ZOMBIE_SIZE/2, z['pos'][1] - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
                if proj_rect.colliderect(zombie_rect):
                    p['hit'] = True
                    z['health'] -= self.PROJECTILE_DAMAGE
                    z['hit_timer'] = 5 # Flash red for 5 frames
                    # sfx: zombie_hit.wav
                    for _ in range(5):
                        self.particles.append(self._create_particle(z['pos'], color=self.COLOR_ZOMBIE, lifespan=10))

                    if z['health'] <= 0:
                        self.zombies.remove(z)
                        self.score += 10
                        reward += 1 # Kill reward
                        # sfx: zombie_die.wav
                        for _ in range(20):
                            self.particles.append(self._create_particle(z['pos'], color=self.COLOR_ZOMBIE, lifespan=20, speed_mult=2.0))

                    if p in self.projectiles:
                        self.projectiles.remove(p)
                    break

        # Zombies vs Player
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for z in self.zombies:
            zombie_rect = pygame.Rect(z['pos'][0] - self.ZOMBIE_SIZE/2, z['pos'][1] - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect):
                self.player_health -= self.ZOMBIE_DAMAGE
                self.player_health = max(0, self.player_health)
                # sfx: player_hit.wav

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particle(self, pos, color, lifespan, speed_mult=1.0, base_vel=np.array([0,0])):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, 2.0) * speed_mult
        vel = base_vel + np.array([math.cos(angle), math.sin(angle)]) * speed
        return {'pos': pos.copy(), 'vel': vel, 'life': lifespan, 'color': color}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / self.PARTICLE_LIFESPAN))
            color = p['color']
            pygame.draw.circle(self.screen, color + (alpha,) if len(color) == 3 else color, p['pos'].astype(int), int(p['life']/4))

        # Zombies
        for z in self.zombies:
            pos = z['pos'].astype(int)
            size = self.ZOMBIE_SIZE
            color = (255, 50, 50) if z['hit_timer'] > 0 else self.COLOR_ZOMBIE
            
            # Glow effect
            glow_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_ZOMBIE_GLOW, (size, size), size)
            self.screen.blit(glow_surf, (pos[0]-size, pos[1]-size))

            pygame.draw.rect(self.screen, color, (pos[0] - size/2, pos[1] - size/2, size, size))

        # Player
        player_pos_int = self.player_pos.astype(int)
        size = self.PLAYER_SIZE
        
        # Glow effect
        glow_surf = pygame.Surface((size*4, size*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (size*2, size*2), size*2)
        self.screen.blit(glow_surf, (player_pos_int[0]-size*2, player_pos_int[1]-size*2))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (player_pos_int[0] - size/2, player_pos_int[1] - size/2, size, size))
        
        # Aiming indicator
        aim_end = self.player_pos + self.last_move_direction * (size * 1.5)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, player_pos_int, aim_end.astype(int), 2)

        # Projectiles
        for p in self.projectiles:
            start_pos = p['pos'] - p['dir'] * self.PROJECTILE_HEIGHT / 2
            end_pos = p['pos'] + p['dir'] * self.PROJECTILE_HEIGHT / 2
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos.astype(int), end_pos.astype(int), self.PROJECTILE_WIDTH)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_ratio), bar_height))

        # Ammo Count
        ammo_text = self.font_medium.render(f"AMMO: {self.ammo}/{self.PLAYER_MAX_AMMO}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (self.WIDTH - ammo_text.get_width() - 10, 10))

        # Timer
        time_str = f"{self.time_left // self.FPS:02d}"
        time_text = self.font_medium.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH/2 - time_text.get_width()/2, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH/2 - score_text.get_width()/2, 45))

        # Reloading Indicator
        if self.reload_timer > 0:
            reload_text = self.font_medium.render("RELOADING...", True, self.COLOR_RELOAD)
            self.screen.blit(reload_text, (self.WIDTH/2 - reload_text.get_width()/2, self.HEIGHT - 50))

        # Game Over / Win Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = self.font_large.render("YOU SURVIVED", True, (50, 255, 50))
            else:
                end_text = self.font_large.render("GAME OVER", True, (255, 50, 50))
            
            final_score_text = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)

            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - 50))
            self.screen.blit(final_score_text, (self.WIDTH/2 - final_score_text.get_width()/2, self.HEIGHT/2 + 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "health": self.player_health,
            "ammo": self.ammo,
            "zombies": len(self.zombies)
        }

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

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless

    env = GameEnv(render_mode="rgb_array")
    
    # --- Test Reset ---
    obs, info = env.reset()
    print("Reset successful.")
    print(f"Initial Info: {info}")

    # --- Test a few steps with random actions ---
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i + 1) % 20 == 0:
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print(f"Episode terminated at step {i+1}.")
            break
    
    print(f"\nTotal reward after 100 random steps: {total_reward:.2f}")
    env.close()

    # --- To visualize the game (requires a display) ---
    print("\nTo run with visualization, comment out the 'os.environ' line and run the following code:")
    # env = GameEnv(render_mode="human")
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    # pygame.display.set_caption("Zombie Survival")
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False

    #     # Simple keyboard mapping for human play
    #     keys = pygame.key.get_pressed()
    #     movement = 0 # none
    #     if keys[pygame.K_UP]: movement = 1
    #     elif keys[pygame.K_DOWN]: movement = 2
    #     elif keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
        
    #     space_held = 1 if keys[pygame.K_SPACE] else 0
    #     shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
    #     action = [movement, space_held, shift_held]
        
    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     # Render the observation to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}")
    #         pygame.time.wait(3000) # Pause for 3 seconds
    #         obs, info = env.reset()

    #     env.clock.tick(GameEnv.FPS)
    # env.close()