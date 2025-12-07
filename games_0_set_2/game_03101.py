
# Generated: 2025-08-28T06:58:48.014101
# Source Brief: brief_03101.md
# Brief Index: 3101

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys for isometric movement. Hold space to sprint. "
        "Survive the horde and reach the green helicopter pad."
    )

    game_description = (
        "Escape hordes of procedurally generated zombies in an isometric 2D world and reach the rescue helicopter. "
        "Sprinting consumes stamina, which regenerates when you stand still. You can withstand 5 bites before you are overwhelmed."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and World Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = 2000
        self.WORLD_HEIGHT = 2000

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (40, 43, 48)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0, 50)
        self.COLOR_ZOMBIE = (220, 50, 50)
        self.COLOR_ZOMBIE_GLOW = (220, 50, 50, 50)
        self.COLOR_HELI = (50, 220, 50)
        self.COLOR_HELI_GLOW = (50, 220, 50, 70)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_STAMINA_BAR = (0, 150, 255)
        self.COLOR_STAMINA_BG = (60, 60, 60)

        # Game constants
        self.FPS = 30
        self.MAX_EPISODE_STEPS = 30 * 120 # 2 minutes
        
        self.PLAYER_RADIUS = 12
        self.PLAYER_SPEED = 4.0
        self.PLAYER_SPRINT_MULTIPLIER = 1.8
        self.MAX_STAMINA = 100
        self.STAMINA_DEPLETION = 1.5
        self.STAMINA_REGEN = 1.0

        self.ZOMBIE_RADIUS = 10
        self.ZOMBIE_BASE_SPEED = 1.5
        self.ZOMBIE_SPAWN_INTERVAL_START = 5 * self.FPS
        self.ZOMBIE_SPAWN_RATE_INCREASE_INTERVAL = 30 * self.FPS
        self.ZOMBIE_SPEED_INCREASE_INTERVAL = 60 * self.FPS

        self.HELI_RADIUS = 40
        self.BITE_COOLDOWN_FRAMES = 1 * self.FPS
        self.MAX_BITES = 5
        
        # State variables will be initialized in reset()
        self.player_pos = None
        self.player_stamina = None
        self.bite_count = None
        self.bite_cooldown = None
        
        self.zombies = None
        self.zombie_speed = None
        self.zombie_spawn_timer = None
        self.zombie_spawn_interval = None
        
        self.heli_pos = None
        self.particles = None
        
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=np.float32)
        self.player_stamina = self.MAX_STAMINA
        self.bite_count = 0
        self.bite_cooldown = 0
        
        self.zombies = []
        self.zombie_speed = self.ZOMBIE_BASE_SPEED
        self.zombie_spawn_timer = 0
        self.zombie_spawn_interval = self.ZOMBIE_SPAWN_INTERVAL_START
        
        # Place helicopter far from player
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist = self.np_random.uniform(self.WORLD_WIDTH * 0.3, self.WORLD_WIDTH * 0.45)
        self.heli_pos = np.array([
            self.player_pos[0] + dist * math.cos(angle),
            self.player_pos[1] + dist * math.sin(angle)
        ], dtype=np.float32)
        self.heli_pos = np.clip(self.heli_pos, [self.HELI_RADIUS, self.HELI_RADIUS], [self.WORLD_WIDTH - self.HELI_RADIUS, self.WORLD_HEIGHT - self.HELI_RADIUS])

        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        
        reward = 0.1 / self.FPS # Survival reward

        # --- Handle Input & Player Movement ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        move_vec = np.array([0.0, 0.0])
        is_moving = False
        if movement == 1: # Up -> Up-Left
            move_vec = np.array([-1.0, -0.5])
            is_moving = True
        elif movement == 2: # Down -> Down-Right
            move_vec = np.array([1.0, 0.5])
            is_moving = True
        elif movement == 3: # Left -> Down-Left
            move_vec = np.array([-1.0, 0.5])
            is_moving = True
        elif movement == 4: # Right -> Up-Right
            move_vec = np.array([1.0, -0.5])
            is_moving = True
        
        if np.linalg.norm(move_vec) > 0:
            move_vec /= np.linalg.norm(move_vec)

        speed = self.PLAYER_SPEED
        if space_held and self.player_stamina > 0 and is_moving:
            speed *= self.PLAYER_SPRINT_MULTIPLIER
            self.player_stamina = max(0, self.player_stamina - self.STAMINA_DEPLETION)
        elif not is_moving:
            self.player_stamina = min(self.MAX_STAMINA, self.player_stamina + self.STAMINA_REGEN)

        self.player_pos += move_vec * speed
        self.player_pos = np.clip(self.player_pos, [0, 0], [self.WORLD_WIDTH, self.WORLD_HEIGHT])

        # --- Update Game State ---
        self.bite_cooldown = max(0, self.bite_cooldown - 1)
        
        # Difficulty scaling
        if self.steps % self.ZOMBIE_SPAWN_RATE_INCREASE_INTERVAL == 0 and self.steps > 0:
            self.zombie_spawn_interval = max(0.5 * self.FPS, self.zombie_spawn_interval * 0.8)
        if self.steps % self.ZOMBIE_SPEED_INCREASE_INTERVAL == 0 and self.steps > 0:
            self.zombie_speed += 0.05
            
        # Zombie spawning
        self.zombie_spawn_timer += 1
        if self.zombie_spawn_timer >= self.zombie_spawn_interval:
            self.zombie_spawn_timer = 0
            self._spawn_zombie()
            
        # Update zombies
        for z in self.zombies:
            direction = self.player_pos - z['pos']
            dist = np.linalg.norm(direction)
            if dist > 1:
                direction /= dist
            z['pos'] += direction * self.zombie_speed
            
            # Zombie-player collision
            if dist < self.PLAYER_RADIUS + self.ZOMBIE_RADIUS and self.bite_cooldown == 0:
                self.bite_count += 1
                self.bite_cooldown = self.BITE_COOLDOWN_FRAMES
                reward -= 1.0
                self.score -= 1
                # sfx: player_hurt.wav
                for _ in range(20):
                    self._create_particle(self.player_pos, self.COLOR_ZOMBIE, 30)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # --- Check Termination ---
        terminated = False
        dist_to_heli = np.linalg.norm(self.player_pos - self.heli_pos)
        if dist_to_heli < self.PLAYER_RADIUS + self.HELI_RADIUS:
            terminated = True
            self.game_over = True
            reward += 10.0
            self.score += 10
            # sfx: victory.wav
            if self.bite_count == 0:
                reward += 50.0
                self.score += 50
        
        if self.bite_count >= self.MAX_BITES:
            terminated = True
            self.game_over = True
            # sfx: game_over.wav
            
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_zombie(self):
        # Spawn off-screen
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            x = self.np_random.uniform(self.player_pos[0] - self.SCREEN_WIDTH, self.player_pos[0] + self.SCREEN_WIDTH)
            y = self.player_pos[1] - self.SCREEN_HEIGHT / 2 - 50
        elif edge == 1: # Bottom
            x = self.np_random.uniform(self.player_pos[0] - self.SCREEN_WIDTH, self.player_pos[0] + self.SCREEN_WIDTH)
            y = self.player_pos[1] + self.SCREEN_HEIGHT / 2 + 50
        elif edge == 2: # Left
            x = self.player_pos[0] - self.SCREEN_WIDTH / 2 - 50
            y = self.np_random.uniform(self.player_pos[1] - self.SCREEN_HEIGHT, self.player_pos[1] + self.SCREEN_HEIGHT)
        else: # Right
            x = self.player_pos[0] + self.SCREEN_WIDTH / 2 + 50
            y = self.np_random.uniform(self.player_pos[1] - self.SCREEN_HEIGHT, self.player_pos[1] + self.SCREEN_HEIGHT)
            
        pos = np.array([x, y], dtype=np.float32)
        self.zombies.append({'pos': pos})

    def _create_particle(self, pos, color, lifetime):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': lifetime, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Camera centered on player
        cam_x = self.player_pos[0] - self.SCREEN_WIDTH / 2
        cam_y = self.player_pos[1] - self.SCREEN_HEIGHT / 2

        # Render background grid
        grid_size = 50
        start_x = int(-cam_x % grid_size)
        start_y = int(-cam_y % grid_size)
        for x in range(start_x, self.SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(start_y, self.SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (int(y), self.SCREEN_WIDTH))

        # Collect all drawable objects
        drawables = []
        
        # Helicopter
        drawables.append({'pos': self.heli_pos, 'y': self.heli_pos[1], 'type': 'heli'})
        
        # Zombies
        for z in self.zombies:
            drawables.append({'pos': z['pos'], 'y': z['pos'][1], 'type': 'zombie'})
            
        # Player
        drawables.append({'pos': self.player_pos, 'y': self.player_pos[1], 'type': 'player'})
        
        # Sort by y-coordinate for isometric rendering
        drawables.sort(key=lambda d: d['y'])

        # Render sorted objects
        for d in drawables:
            screen_pos = (int(d['pos'][0] - cam_x), int(d['pos'][1] - cam_y))
            
            if d['type'] == 'player':
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], self.PLAYER_RADIUS + 5, self.COLOR_PLAYER_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
                pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            
            elif d['type'] == 'zombie':
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], self.ZOMBIE_RADIUS + 4, self.COLOR_ZOMBIE_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], self.ZOMBIE_RADIUS, self.COLOR_ZOMBIE)
                pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], self.ZOMBIE_RADIUS, self.COLOR_ZOMBIE)
            
            elif d['type'] == 'heli':
                # Pad
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], self.HELI_RADIUS + 10, self.COLOR_HELI_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], self.HELI_RADIUS, self.COLOR_HELI)
                pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], self.HELI_RADIUS, self.COLOR_HELI)
                # 'H'
                h_text = self.font_large.render('H', True, self.COLOR_BG)
                self.screen.blit(h_text, (screen_pos[0] - h_text.get_width() / 2, screen_pos[1] - h_text.get_height() / 2))
                # Blades
                angle = (self.steps * 15) % 360
                for i in range(2):
                    rad = math.radians(angle + i * 180)
                    end_x = screen_pos[0] + math.cos(rad) * (self.HELI_RADIUS * 1.5)
                    end_y = screen_pos[1] + math.sin(rad) * (self.HELI_RADIUS * 1.5)
                    pygame.draw.line(self.screen, (10,10,10), screen_pos, (int(end_x), int(end_y)), 6)


        # Render particles
        for p in self.particles:
            screen_pos = (int(p['pos'][0] - cam_x), int(p['pos'][1] - cam_y))
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], 2, color)

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Bite Counter
        bite_text = self.font_small.render(f"Bites: {self.bite_count} / {self.MAX_BITES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(bite_text, (10, 10))
        
        # Timer
        seconds = self.steps // self.FPS
        timer_text = self.font_small.render(f"Time: {seconds}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Stamina Bar
        stamina_bar_width = 200
        stamina_bar_height = 15
        stamina_x = (self.SCREEN_WIDTH - stamina_bar_width) / 2
        stamina_y = self.SCREEN_HEIGHT - stamina_bar_height - 10
        
        fill_ratio = self.player_stamina / self.MAX_STAMINA
        pygame.draw.rect(self.screen, self.COLOR_STAMINA_BG, (stamina_x, stamina_y, stamina_bar_width, stamina_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_STAMINA_BAR, (stamina_x, stamina_y, stamina_bar_width * fill_ratio, stamina_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (stamina_x, stamina_y, stamina_bar_width, stamina_bar_height), 1)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.bite_count >= self.MAX_BITES:
                msg = "OVERWHELMED"
            elif np.linalg.norm(self.player_pos - self.heli_pos) < self.PLAYER_RADIUS + self.HELI_RADIUS:
                msg = "RESCUED!"
            else:
                msg = "TIME'S UP"
                
            game_over_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(game_over_text, (self.SCREEN_WIDTH/2 - game_over_text.get_width()/2, self.SCREEN_HEIGHT/2 - game_over_text.get_height()/2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bite_count": self.bite_count,
            "stamina": self.player_stamina,
            "player_pos": self.player_pos,
            "heli_pos": self.heli_pos,
            "zombie_count": len(self.zombies),
        }

    def close(self):
        pygame.font.quit()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Requires pygame to be installed with video support
    import os
    # if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
    #     print("Cannot run interactive test in dummy video mode. Exiting.")
    #     exit()
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    
    terminated = False
    total_reward = 0
    
    # Mapping from Pygame keys to action components
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not terminated:
        # --- Human Input ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        for key, move in key_to_action.items():
            if keys[key]:
                movement_action = move
                break # Prioritize first key found
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
        if keys[pygame.K_ESCAPE]:
            terminated = True
            
        action = [movement_action, space_action, shift_action]
        
        # --- Environment Step ---
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term or trunc
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()