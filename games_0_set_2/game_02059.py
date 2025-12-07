
# Generated: 2025-08-28T03:34:00.165378
# Source Brief: brief_02059.md
# Brief Index: 2059

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A top-down, wave-based tower defense game. The player places defensive barriers
    to protect a group of stationary survivors from incoming hordes of zombies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Hold Space to place a Wall. "
        "Hold Space + Shift to place Spikes. Place a Wall on Spikes (or vice-versa) to create an explosion."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of zombies by strategically placing defensive barriers. "
        "Survive all 5 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.FPS = 30
        
        # --- Game Constants ---
        self.MAX_STEPS = 4500 # 150 seconds at 30 FPS
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_BASE = (40, 45, 50)
        self.COLOR_SURVIVOR = (0, 255, 128)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_BARRIER_WALL = (139, 69, 19)
        self.COLOR_BARRIER_SPIKE = (100, 100, 120)
        self.COLOR_EXPLOSION = (255, 165, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_GREEN = (0, 200, 0)
        self.COLOR_HEALTH_RED = (200, 0, 0)
        self.COLOR_SCRAP = (255, 215, 0)

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_wave = pygame.font.SysFont("monospace", 48, bold=True)

        # Game parameters
        self.BASE_Y_START = 320
        self.SURVIVOR_COUNT = 10
        self.WAVE_CONFIG = [
            {"count": 10, "speed": 0.8, "health": 100},
            {"count": 15, "speed": 1.0, "health": 120},
            {"count": 20, "speed": 1.2, "health": 140},
            {"count": 25, "speed": 1.4, "health": 160},
            {"count": 30, "speed": 1.6, "health": 180},
        ]
        self.BARRIER_SPECS = {
            1: {"cost": 10, "health": 300, "damage": 0, "color": self.COLOR_BARRIER_WALL, "name": "WALL"}, # Wall
            2: {"cost": 25, "health": 150, "damage": 2, "color": self.COLOR_BARRIER_SPIKE, "name": "SPIKE"}, # Spike
        }
        self.COMBO_EXPLOSION_RADIUS = 60
        self.COMBO_EXPLOSION_DAMAGE = 250
        self.COMBO_SCRAP_BONUS = 20

        # Initialize state variables (to be defined in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.survivors = None
        self.zombies = None
        self.barriers = None
        self.particles = None
        self.cursor_pos = None
        self.wave = None
        self.scrap = None
        self.wave_transition_timer = None
        
        # Initialize state
        self.reset()
        
        # CRITICAL: Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        self.scrap = 50
        self.wave_transition_timer = self.FPS * 2 # Initial countdown

        self.survivors = self._spawn_survivors()
        self.zombies = []
        self.barriers = []
        self.particles = []
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # Handle wave transitions
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer == 0 and self.wave <= len(self.WAVE_CONFIG):
                self._start_wave()
        else:
            # Only update game logic if not in transition
            step_reward = self._update_game_logic(action)
            reward += step_reward

        # Check for termination conditions
        survivors_alive = sum(1 for s in self.survivors if s['health'] > 0)
        if survivors_alive == 0:
            terminated = True
            self.game_over = True
            reward = -100 # Losing penalty
        elif self.wave > len(self.WAVE_CONFIG) and not self.zombies:
            terminated = True
            self.game_over = True
            reward += 50 # Winning bonus
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_logic(self, action):
        reward = 0
        
        self._handle_input(action)
        reward += self._update_zombies()
        self._update_barriers()
        self._update_particles()
        
        # Check for wave completion
        if not self.zombies and self.wave <= len(self.WAVE_CONFIG):
            reward += 10 # Wave complete reward
            self.scrap += 75 # Bonus scrap between waves
            self.wave += 1
            if self.wave <= len(self.WAVE_CONFIG):
                self.wave_transition_timer = self.FPS * 3 # 3 second pause
            
        return reward

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        cursor_speed = 10
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 40, self.BASE_Y_START - 10)

        if space_held:
            barrier_type = 2 if shift_held else 1
            self._place_barrier(barrier_type)

    def _place_barrier(self, barrier_type):
        spec = self.BARRIER_SPECS[barrier_type]
        if self.scrap < spec["cost"]:
            return # Not enough scrap

        # Check for combo: placing on an existing barrier of the other type
        for i, b in reversed(list(enumerate(self.barriers))):
            dist = np.linalg.norm(self.cursor_pos - b['pos'])
            if dist < 20 and b['type'] != barrier_type:
                self.scrap -= spec["cost"] # Pay the cost
                self.scrap += self.COMBO_SCRAP_BONUS
                self._create_explosion(b['pos'], self.COMBO_EXPLOSION_RADIUS, self.COMBO_EXPLOSION_DAMAGE)
                # SFX: Big explosion
                del self.barriers[i] # Remove the existing barrier
                return

        # Check for placing too close to another barrier of the same type
        for b in self.barriers:
            if np.linalg.norm(self.cursor_pos - b['pos']) < 30:
                return # Too close

        # Place new barrier
        self.scrap -= spec["cost"]
        self.barriers.append({
            'pos': self.cursor_pos.copy(),
            'type': barrier_type,
            'health': spec["health"],
            'max_health': spec["health"],
        })
        # SFX: place_wood.wav or place_metal.wav

    def _update_zombies(self):
        reward = 0
        zombies_to_remove = []
        alive_survivors = [s for s in self.survivors if s['health'] > 0]

        for i, zombie in enumerate(self.zombies):
            if not alive_survivors: break
            
            target = min(alive_survivors, key=lambda s: np.linalg.norm(zombie['pos'] - s['pos']))
            direction = target['pos'] - zombie['pos']
            dist_to_target = np.linalg.norm(direction)
            
            # Check collision with barriers
            is_blocked = False
            for barrier in self.barriers:
                if np.linalg.norm(zombie['pos'] - barrier['pos']) < 25:
                    is_blocked = True
                    barrier['health'] -= 1
                    zombie['health'] -= self.BARRIER_SPECS[barrier['type']]['damage']
                    if self.BARRIER_SPECS[barrier['type']]['damage'] > 0:
                        self._create_hit_particle(zombie['pos'])
                    break
            
            # Move if not blocked
            if not is_blocked and dist_to_target > 1:
                zombie['pos'] += (direction / dist_to_target) * zombie['speed']

            # Check collision with base/survivors
            if zombie['pos'][1] >= self.BASE_Y_START - 5:
                survivor_damage = 1
                target['health'] -= survivor_damage
                reward -= 0.1 * survivor_damage
                if target['health'] <= 0:
                    # SFX: survivor_death.wav
                    alive_survivors = [s for s in self.survivors if s['health'] > 0]

            if zombie['health'] <= 0:
                zombies_to_remove.append(i)
                reward += 0.1
                self.scrap += 5
                self._create_death_particles(zombie['pos'])
                # SFX: zombie_death.wav
        
        for i in sorted(zombies_to_remove, reverse=True):
            del self.zombies[i]
        return reward
        
    def _update_barriers(self):
        self.barriers = [b for b in self.barriers if b['health'] > 0]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.95

    def _spawn_survivors(self):
        return [{'pos': np.array([self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(self.BASE_Y_START + 20, self.HEIGHT - 20)]),
                 'health': 100, 'max_health': 100} for _ in range(self.SURVIVOR_COUNT)]
        
    def _start_wave(self):
        wave_info = self.WAVE_CONFIG[self.wave - 1]
        for _ in range(wave_info['count']):
            self.zombies.append({
                'pos': np.array([self.np_random.uniform(20, self.WIDTH - 20), -20 - self.np_random.uniform(0, 100)]),
                'health': wave_info['health'], 'max_health': wave_info['health'],
                'speed': wave_info['speed']
            })

    def _create_explosion(self, pos, radius, damage):
        for z in self.zombies:
            if np.linalg.norm(z['pos'] - pos) < radius:
                z['health'] -= damage
        for _ in range(70):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 10)
            self.particles.append({
                'pos': pos.copy(), 'vel': np.array([math.cos(angle), math.sin(angle)]) * speed,
                'lifespan': self.np_random.integers(20, 40), 'radius': self.np_random.uniform(3, 7),
                'color': random.choice([self.COLOR_EXPLOSION, (255, 69, 0), (255, 255, 0)])
            })

    def _create_death_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(), 'vel': np.array([math.cos(angle), math.sin(angle)]) * speed,
                'lifespan': self.np_random.integers(10, 20), 'radius': self.np_random.uniform(1, 3),
                'color': self.COLOR_ZOMBIE
            })
            
    def _create_hit_particle(self, pos):
        self.particles.append({
            'pos': pos.copy() + self.np_random.uniform(-5, 5, 2), 'vel': np.array([0, 0]),
            'lifespan': 5, 'radius': 3, 'color': (200, 200, 200)
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_BASE, (0, self.BASE_Y_START, self.WIDTH, self.HEIGHT - self.BASE_Y_START))
        for b in self.barriers: self._render_barrier(b)
        for s in self.survivors:
            if s['health'] > 0: self._render_survivor(s)
        for z in self.zombies: self._render_zombie(z)
        for p in self.particles:
            pos = p['pos'].astype(int)
            pygame.draw.circle(self.screen, p['color'], pos, max(1, int(p['radius'])))
        self._render_cursor()

    def _render_survivor(self, s):
        pos = s['pos'].astype(int)
        pygame.draw.circle(self.screen, self.COLOR_SURVIVOR, pos, 8)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_SURVIVOR)
        self._render_health_bar(pos - np.array([15, 15]), 30, s['health'], s['max_health'])

    def _render_zombie(self, z):
        pos = z['pos'].astype(int)
        pygame.draw.circle(self.screen, self.COLOR_ZOMBIE, pos, 10)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, (*self.COLOR_ZOMBIE, 100))
        self._render_health_bar(pos - np.array([15, 20]), 30, z['health'], z['max_health'])

    def _render_barrier(self, b):
        pos = b['pos'].astype(int)
        size = 15
        rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
        color = self.BARRIER_SPECS[b['type']]['color']
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        if b['type'] == 2: # Spikes
            for i in range(-1, 2, 2):
                for j in range(-1, 2, 2):
                    p1 = pos + np.array([i*size, j*size])
                    p2 = pos + np.array([i*size*0.5, j*size*1.5])
                    p3 = pos + np.array([i*size*1.5, j*size*0.5])
                    pygame.draw.polygon(self.screen, (150,150,170), [p1,p2,p3])
        self._render_health_bar(pos - np.array([15, 25]), 30, b['health'], b['max_health'])

    def _render_cursor(self):
        pos = self.cursor_pos.astype(int)
        size = 15
        s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        
        is_combo = any(np.linalg.norm(self.cursor_pos - b['pos']) < 20 for b in self.barriers)
        color = self.COLOR_EXPLOSION if is_combo else (255, 255, 255)
        
        pygame.draw.rect(s, (*color, 100), (0, 0, size*2, size*2), 0, 3)
        pygame.draw.rect(s, (*color, 200), (0, 0, size*2, size*2), 2, 3)
        self.screen.blit(s, (pos[0]-size, pos[1]-size))

    def _render_health_bar(self, pos, width, health, max_health):
        health_ratio = np.clip(health / max_health, 0, 1)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (*pos, width, 5), border_radius=1)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (*pos, int(width * health_ratio), 5), border_radius=1)

    def _render_ui(self):
        pygame.draw.rect(self.screen, (10, 10, 10, 200), (0, 0, self.WIDTH, 40))
        
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 12))
        
        wave_str = f"WAVE: {self.wave}/{len(self.WAVE_CONFIG)}" if self.wave <= len(self.WAVE_CONFIG) else "WAVES CLEAR"
        wave_text = self.font_ui.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (140, 12))
        
        s_alive = sum(1 for s in self.survivors if s['health'] > 0)
        s_color = self.COLOR_SURVIVOR if s_alive > 3 else self.COLOR_ZOMBIE
        survivor_text = self.font_ui.render(f"SURVIVORS: {s_alive}/{self.SURVIVOR_COUNT}", True, s_color)
        self.screen.blit(survivor_text, (280, 12))
        
        scrap_text = self.font_ui.render(f"SCRAP: {self.scrap}", True, self.COLOR_SCRAP)
        self.screen.blit(scrap_text, (440, 12))

        # Show selected item
        keys = pygame.key.get_pressed()
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        barrier_type = 2 if shift_held else 1
        spec = self.BARRIER_SPECS[barrier_type]
        sel_text = self.font_ui.render(f"PLACE: {spec['name']} (${spec['cost']})", True, spec['color'])
        self.screen.blit(sel_text, (520, 12))

        if self.wave_transition_timer > 0:
            msg = f"WAVE {self.wave-1} COMPLETE" if self.wave > 1 and self.wave <= len(self.WAVE_CONFIG) else ""
            if self.wave > len(self.WAVE_CONFIG): msg = "YOU WIN!"
            
            if msg:
                surf = self.font_wave.render(msg, True, self.COLOR_TEXT)
                self.screen.blit(surf, (self.WIDTH/2 - surf.get_width()/2, self.HEIGHT/2 - 80))
            
            if self.wave <= len(self.WAVE_CONFIG):
                next_msg = f"WAVE {self.wave} IN {self.wave_transition_timer / self.FPS:.1f}s"
                next_surf = self.font_ui.render(next_msg, True, self.COLOR_TEXT)
                self.screen.blit(next_surf, (self.WIDTH/2 - next_surf.get_width()/2, self.HEIGHT/2 - 20))
            
        if self.game_over and s_alive == 0:
            surf = self.font_wave.render("GAME OVER", True, self.COLOR_ZOMBIE)
            self.screen.blit(surf, (self.WIDTH/2 - surf.get_width()/2, self.HEIGHT/2 - 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "survivors_left": sum(1 for s in self.survivors if s['health'] > 0),
            "zombies_left": len(self.zombies),
            "scrap": self.scrap,
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping from keyboard to MultiDiscrete ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Game Loop Control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}. Info: {info}")
            # Wait for a moment before auto-resetting, or wait for 'R' key
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()