
# Generated: 2025-08-28T00:45:37.715880
# Source Brief: brief_03894.md
# Brief Index: 3894

        
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
        "Controls: Arrow keys to move. Hold Space to shoot. Press Shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for 5 minutes against an ever-growing horde of zombies in a top-down arena shooter."
    )

    # Should frames auto-advance or wait for user input?
    # Brief: "Turn-based, each step represents a fixed time interval." -> auto_advance = False
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CONSTANTS ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.ARENA_MARGIN = 10

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (80, 80, 90)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_AMMO_PACK = (50, 200, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_HIGH = (0, 255, 0)
        self.COLOR_HEALTH_MED = (255, 255, 0)
        self.COLOR_HEALTH_LOW = (255, 0, 0)
        self.COLOR_HEALTH_BG = (80, 0, 0)
        self.COLOR_MUZZLE_FLASH = (255, 235, 150)
        self.COLOR_AIM_LINE = (100, 100, 110)

        # Game parameters
        self.MAX_STEPS = 3000
        self.PLAYER_SPEED = 5
        self.PLAYER_SIZE = 12
        self.MAX_HEALTH = 100
        self.MAX_AMMO = 30
        self.RELOAD_TIME = 10
        self.PROJECTILE_SPEED = 15
        self.PROJECTILE_SIZE = 3
        self.ZOMBIE_SPEED = 1.5
        self.ZOMBIE_SIZE = 12
        self.ZOMBIE_INITIAL_SPAWN_RATE = 0.03
        self.ZOMBIE_SPAWN_RAMP = 0.01 / 100
        self.AMMO_PACK_SIZE = 10
        self.AMMO_PACK_VALUE = 15
        self.AMMO_SPAWN_RATE = 0.005
        
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_aim_direction = None
        self.reloading_timer = None
        self.zombies = None
        self.projectiles = None
        self.ammo_packs = None
        self.particles = None
        self.steps = None
        self.kill_count = None
        self.game_over = None
        self.np_random = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.np_random = np.random.default_rng(seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.MAX_HEALTH
        self.player_ammo = self.MAX_AMMO
        self.player_aim_direction = np.array([0, -1], dtype=np.float32) # Aim up
        self.reloading_timer = 0
        
        self.zombies = []
        self.projectiles = []
        self.ammo_packs = []
        self.particles = []

        self.steps = 0
        self.kill_count = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1 # Survival reward
        self.steps += 1

        self._handle_player_action(action)
        self._update_projectiles()
        reward += self._update_zombies()
        self._update_ammo_packs()
        self._update_particles()
        self._spawn_entities()
        
        terminated = (self.player_health <= 0) or (self.steps >= self.MAX_STEPS)
        if terminated:
            self.game_over = True
            if self.player_health > 0 and self.steps >= self.MAX_STEPS:
                reward += 100 # Victory bonus

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle reloading state
        if self.reloading_timer > 0:
            self.reloading_timer -= 1
            if self.reloading_timer == 0:
                self.player_ammo = self.MAX_AMMO
                # SFX: Reload complete
            return # No other actions while reloading

        # Handle movement
        move_vector = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vector[1] -= 1 # Up
        elif movement == 2: move_vector[1] += 1 # Down
        elif movement == 3: move_vector[0] -= 1 # Left
        elif movement == 4: move_vector[0] += 1 # Right

        if np.any(move_vector):
            # Normalize for consistent speed
            norm = np.linalg.norm(move_vector)
            if norm > 0:
                move_vector /= norm
                self.player_pos += move_vector * self.PLAYER_SPEED
                self.player_aim_direction = move_vector # Update aim direction
        
        # Clamp player position
        self.player_pos[0] = np.clip(self.player_pos[0], self.ARENA_MARGIN + self.PLAYER_SIZE/2, self.WIDTH - self.ARENA_MARGIN - self.PLAYER_SIZE/2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.ARENA_MARGIN + self.PLAYER_SIZE/2, self.HEIGHT - self.ARENA_MARGIN - self.PLAYER_SIZE/2)

        # Handle shooting
        if space_held and self.player_ammo > 0:
            self.player_ammo -= 1
            proj_pos = self.player_pos + self.player_aim_direction * (self.PLAYER_SIZE / 2 + 5)
            self.projectiles.append({
                'pos': proj_pos.copy(),
                'vel': self.player_aim_direction.copy()
            })
            # SFX: Player shoot
            self._create_muzzle_flash()

        # Handle reloading trigger
        if shift_held and self.player_ammo < self.MAX_AMMO:
            self.reloading_timer = self.RELOAD_TIME
            # SFX: Start reload

    def _update_projectiles(self):
        for p in self.projectiles:
            p['pos'] += p['vel'] * self.PROJECTILE_SPEED
        # Remove off-screen projectiles
        self.projectiles = [p for p in self.projectiles if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT]

    def _update_zombies(self):
        kill_reward = 0
        zombies_to_keep = []
        projectiles_to_keep = self.projectiles[:]

        for z in self.zombies:
            # Move towards player
            direction = self.player_pos - z['pos']
            norm = np.linalg.norm(direction)
            if norm > 0:
                z['pos'] += (direction / norm) * self.ZOMBIE_SPEED
            
            # Check collision with player
            if np.linalg.norm(self.player_pos - z['pos']) < (self.PLAYER_SIZE + self.ZOMBIE_SIZE) / 2:
                self.player_health -= 10
                # SFX: Player hit
                kill_reward -= 0.1 # Hit penalty
                continue # Zombie is consumed on hit

            # Check collision with projectiles
            hit_by_projectile = False
            for p in projectiles_to_keep:
                if np.linalg.norm(p['pos'] - z['pos']) < (self.PROJECTILE_SIZE + self.ZOMBIE_SIZE) / 2:
                    kill_reward += 1.0
                    self.kill_count += 1
                    projectiles_to_keep.remove(p)
                    self._create_blood_splatter(z['pos'])
                    # SFX: Zombie hit/death
                    hit_by_projectile = True
                    break
            
            if not hit_by_projectile:
                # Check despawn distance
                if np.linalg.norm(self.player_pos - z['pos']) < 800: # Despawn if too far
                    zombies_to_keep.append(z)

        self.zombies = zombies_to_keep
        self.projectiles = projectiles_to_keep
        return kill_reward

    def _update_ammo_packs(self):
        packs_to_keep = []
        for pack in self.ammo_packs:
            if np.linalg.norm(self.player_pos - pack['pos']) < (self.PLAYER_SIZE + self.AMMO_PACK_SIZE) / 2:
                self.player_ammo = min(self.MAX_AMMO, self.player_ammo + self.AMMO_PACK_VALUE)
                # SFX: Ammo pickup
            else:
                packs_to_keep.append(pack)
        self.ammo_packs = packs_to_keep
        
    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _spawn_entities(self):
        # Spawn zombies
        spawn_rate = self.ZOMBIE_INITIAL_SPAWN_RATE + self.steps * self.ZOMBIE_SPAWN_RAMP
        if self.np_random.random() < spawn_rate:
            self._spawn_zombie()
        
        # Spawn ammo packs
        if self.np_random.random() < self.AMMO_SPAWN_RATE:
            if len(self.ammo_packs) < 3: # Limit max packs on screen
                self._spawn_ammo_pack()

    def _spawn_zombie(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE])
        elif edge == 2: # Left
            pos = np.array([-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)])
        else: # Right
            pos = np.array([self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)])
        
        # Ensure not too close to player initially
        if np.linalg.norm(pos - self.player_pos) > 200:
            self.zombies.append({'pos': pos})

    def _spawn_ammo_pack(self):
        pos = self.np_random.uniform(
            [self.ARENA_MARGIN + 20, self.ARENA_MARGIN + 20],
            [self.WIDTH - self.ARENA_MARGIN - 20, self.HEIGHT - self.ARENA_MARGIN - 20]
        )
        self.ammo_packs.append({'pos': np.array(pos)})

    def _create_muzzle_flash(self):
        flash_pos = self.player_pos + self.player_aim_direction * (self.PLAYER_SIZE / 2 + 10)
        self.particles.append({
            'pos': flash_pos, 'vel': np.array([0,0]), 'lifetime': 2, 'type': 'flash'
        })

    def _create_blood_splatter(self, pos):
        for _ in range(self.np_random.integers(8, 15)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(5, 15),
                'type': 'blood'
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Arena walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), self.ARENA_MARGIN)

        # Ammo packs
        for pack in self.ammo_packs:
            pygame.draw.rect(self.screen, self.COLOR_AMMO_PACK, (pack['pos'][0] - self.AMMO_PACK_SIZE/2, pack['pos'][1] - self.AMMO_PACK_SIZE/2, self.AMMO_PACK_SIZE, self.AMMO_PACK_SIZE))

        # Zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, (z['pos'][0] - self.ZOMBIE_SIZE/2, z['pos'][1] - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE))

        # Player aim line
        if self.reloading_timer == 0:
            start_pos = self.player_pos
            end_pos = self.player_pos + self.player_aim_direction * 40
            pygame.draw.aaline(self.screen, self.COLOR_AIM_LINE, start_pos, end_pos, 1)

        # Player
        player_rect = (self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Projectiles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), self.PROJECTILE_SIZE, self.COLOR_PROJECTILE)

        # Particles
        for p in self.particles:
            if p['type'] == 'blood':
                alpha = int(255 * (p['lifetime'] / 15))
                color = (*self.COLOR_ZOMBIE, alpha)
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (2, 2), 2)
                self.screen.blit(temp_surf, p['pos'] - np.array([2,2]))
            elif p['type'] == 'flash':
                size = 15 * (p['lifetime'] / 2)
                pos = p['pos']
                pygame.draw.circle(self.screen, self.COLOR_MUZZLE_FLASH, pos.astype(int), int(size))


    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.MAX_HEALTH)
        health_color = self.COLOR_HEALTH_LOW
        if health_ratio > 0.66: health_color = self.COLOR_HEALTH_HIGH
        elif health_ratio > 0.33: health_color = self.COLOR_HEALTH_MED
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (20, 20, 200, 20))
        if health_ratio > 0:
            pygame.draw.rect(self.screen, health_color, (20, 20, 200 * health_ratio, 20))
        
        # Ammo Count
        ammo_text = self.font_small.render(f"AMMO: {self.player_ammo}/{self.MAX_AMMO}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (230, 21))

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_large.render(f"TIME: {time_left}", True, self.COLOR_ZOMBIE)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(time_text, time_rect)

        # Kill Count
        kill_text = self.font_small.render(f"KILLS: {self.kill_count}", True, self.COLOR_TEXT)
        self.screen.blit(kill_text, (20, 45))

        # Reloading Indicator
        if self.reloading_timer > 0:
            reload_text = self.font_large.render("RELOADING...", True, self.COLOR_AMMO_PACK)
            reload_rect = reload_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 50))
            self.screen.blit(reload_text, reload_rect)
        
        # Game Over / Win message
        if self.game_over:
            if self.player_health <= 0:
                msg = "GAME OVER"
                color = self.COLOR_ZOMBIE
            else:
                msg = "YOU SURVIVED!"
                color = self.COLOR_PLAYER
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "kill_count": self.kill_count,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo
        }

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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial Info:", info)

    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished.")
            break
    
    # Example of saving a frame
    try:
        from PIL import Image
        img = Image.fromarray(obs)
        img.save("game_frame.png")
        print("Saved a sample frame to game_frame.png")
    except ImportError:
        print("PIL/Pillow not installed, skipping frame save.")

    env.close()