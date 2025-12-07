import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:48:46.392778
# Source Brief: brief_02313.md
# Brief Index: 2313
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent defends a cell from invading pathogens.
    The agent controls the cell's internal structure and deploys antibodies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend a biological cell from invading pathogens by manipulating internal walls and deploying antibodies to protect the nucleus."
    user_guide = "Controls: Use arrow keys (↑↓←→) to expand defensive walls. Hold Shift + arrow key to retract walls. Press Space to deploy an antibody."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER_X, CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    CELL_RADIUS = 180
    NUCLEUS_RADIUS = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_CELL_MEMBRANE = (60, 40, 90)
    COLOR_NUCLEUS = (100, 200, 100)
    COLOR_HEALTH_BAR_BG = (50, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 255, 0)
    COLOR_PATHOGEN = (255, 50, 50)
    COLOR_ANTIBODY = (50, 150, 255)
    COLOR_PORTAL = (255, 255, 0)
    COLOR_RESEARCH = (180, 50, 255)
    COLOR_WALL = (200, 255, 200)
    COLOR_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 24, bold=True)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Game state variables
        self.cell_health = 0.0
        self.pathogens = []
        self.antibodies = []
        self.particles = []
        self.portals = []
        self.wall_radii = {}
        self.target_wall_radii = {}
        self.research_progress = 0.0
        self.pathogen_spawn_timer = 0
        self.difficulty_tier = 0
        
        # These will be initialized in reset()
        self.pathogen_base_speed = 0
        self.pathogen_base_health = 0
        self.pathogen_spawn_rate = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cell_health = 100.0
        self.difficulty_tier = 0

        self.pathogen_base_speed = 1.0
        self.pathogen_base_health = 1
        self.pathogen_spawn_rate = 90 # Lower is faster

        self.pathogens = []
        self.antibodies = []
        self.particles = []
        
        self.portals = [
            (self.CENTER_X + self.CELL_RADIUS * 0.7, self.CENTER_Y),
            (self.CENTER_X - self.CELL_RADIUS * 0.7, self.CENTER_Y),
            (self.CENTER_X, self.CENTER_Y + self.CELL_RADIUS * 0.7),
            (self.CENTER_X, self.CENTER_Y - self.CELL_RADIUS * 0.7),
        ]

        initial_radius = self.NUCLEUS_RADIUS + 40
        self.wall_radii = {'up': initial_radius, 'down': initial_radius, 'left': initial_radius, 'right': initial_radius}
        self.target_wall_radii = self.wall_radii.copy()

        self.research_progress = 0.0
        
        # Spawn initial pathogens
        for _ in range(3):
            self._spawn_pathogen()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # --- Handle Actions ---
        self._handle_actions(action)
        
        # --- Update Game State ---
        self._update_walls()
        reward += self._update_pathogens()
        reward += self._update_antibodies()
        self._update_particles()
        self._update_research()
        self._spawn_new_pathogens()
        self._update_difficulty()

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            if self.cell_health <= 0:
                reward -= 100 # Defeat penalty
            elif not self.pathogens:
                reward += 100 # Victory bonus
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Wall manipulation
        if movement > 0:
            direction_map = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
            direction = direction_map[movement]
            
            if shift_held:
                # Shrink wall
                self.target_wall_radii[direction] = max(self.NUCLEUS_RADIUS + 5, self.target_wall_radii[direction] - 10)
            else:
                # Grow wall
                self.target_wall_radii[direction] = min(self.CELL_RADIUS - 10, self.target_wall_radii[direction] + 10)

        # Antibody deployment
        if space_held:
            self._deploy_antibody()
            
    def _update_walls(self):
        for d in self.wall_radii:
            diff = self.target_wall_radii[d] - self.wall_radii[d]
            self.wall_radii[d] += diff * 0.2 # Smooth interpolation

    def _update_pathogens(self):
        reward = 0
        for p in self.pathogens:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            
            # Bounce off cell membrane
            dist_from_center = math.hypot(p['pos'][0] - self.CENTER_X, p['pos'][1] - self.CENTER_Y)
            if dist_from_center > self.CELL_RADIUS - p['size']:
                # Reflect velocity
                nx = (p['pos'][0] - self.CENTER_X) / dist_from_center
                ny = (p['pos'][1] - self.CENTER_Y) / dist_from_center
                dot = p['vel'][0] * nx + p['vel'][1] * ny
                p['vel'] = (p['vel'][0] - 2 * dot * nx, p['vel'][1] - 2 * dot * ny)

            # Bounce off internal walls
            self._handle_wall_collisions(p)

            # Check collision with nucleus
            if dist_from_center < self.NUCLEUS_RADIUS + p['size']:
                damage = 10
                self.cell_health -= damage
                reward -= 0.1 * damage # -1 for 10 health loss
                p['active'] = False
                self._create_particles(p['pos'], self.COLOR_PATHOGEN, 20)
        
        self.pathogens = [p for p in self.pathogens if p['active']]
        return reward

    def _handle_wall_collisions(self, p):
        px, py = p['pos']
        ps = p['size']
        
        # Wall rects
        wall_up = pygame.Rect(self.CENTER_X - self.wall_radii['left'], self.CENTER_Y - self.wall_radii['up'], self.wall_radii['left'] + self.wall_radii['right'], 5)
        wall_down = pygame.Rect(self.CENTER_X - self.wall_radii['left'], self.CENTER_Y + self.wall_radii['down'], self.wall_radii['left'] + self.wall_radii['right'], 5)
        wall_left = pygame.Rect(self.CENTER_X - self.wall_radii['left'], self.CENTER_Y - self.wall_radii['up'], 5, self.wall_radii['up'] + self.wall_radii['down'])
        wall_right = pygame.Rect(self.CENTER_X + self.wall_radii['right'], self.CENTER_Y - self.wall_radii['up'], 5, self.wall_radii['up'] + self.wall_radii['down'])

        pathogen_rect = pygame.Rect(px - ps, py - ps, ps*2, ps*2)

        if pathogen_rect.colliderect(wall_up) and p['vel'][1] < 0:
            p['vel'] = (p['vel'][0], -p['vel'][1] * 0.9)
        if pathogen_rect.colliderect(wall_down) and p['vel'][1] > 0:
            p['vel'] = (p['vel'][0], -p['vel'][1] * 0.9)
        if pathogen_rect.colliderect(wall_left) and p['vel'][0] < 0:
            p['vel'] = (-p['vel'][0] * 0.9, p['vel'][1])
        if pathogen_rect.colliderect(wall_right) and p['vel'][0] > 0:
            p['vel'] = (-p['vel'][0] * 0.9, p['vel'][1])

    def _update_antibodies(self):
        reward = 0
        for a in self.antibodies:
            # Find closest pathogen
            if not self.pathogens:
                a['target'] = None
            elif a['target'] is None or not a['target']['active']:
                a['target'] = min(self.pathogens, key=lambda p: math.hypot(a['pos'][0]-p['pos'][0], a['pos'][1]-p['pos'][1]))
            
            if a['target']:
                # Move towards target
                target_pos = a['target']['pos']
                angle = math.atan2(target_pos[1] - a['pos'][1], target_pos[0] - a['pos'][0])
                a['vel'] = (math.cos(angle) * a['speed'], math.sin(angle) * a['speed'])
                a['pos'] = (a['pos'][0] + a['vel'][0], a['pos'][1] + a['vel'][1])

                # Check collision with target
                if math.hypot(a['pos'][0]-target_pos[0], a['pos'][1]-target_pos[1]) < a['size'] + a['target']['size']:
                    a['active'] = False
                    a['target']['active'] = False
                    self.score += 10
                    reward += 0.1 # Small reward for eliminating pathogen
                    self._create_particles(a['target']['pos'], self.COLOR_PATHOGEN, 30)

            a['lifespan'] -= 1
            if a['lifespan'] <= 0:
                a['active'] = False
        
        self.antibodies = [ab for ab in self.antibodies if ab['active']]
        return reward
        
    def _update_particles(self):
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _update_research(self):
        # Passive research gain
        self.research_progress = min(100, self.research_progress + 0.1)

    def _update_difficulty(self):
        new_tier = self.steps // 200
        if new_tier > self.difficulty_tier:
            self.difficulty_tier = new_tier
            self.pathogen_base_speed += 0.05
            self.pathogen_base_health += 0.05
            self.pathogen_spawn_rate = max(30, self.pathogen_spawn_rate - 5)

    def _spawn_new_pathogens(self):
        self.pathogen_spawn_timer -= 1
        if self.pathogen_spawn_timer <= 0:
            self._spawn_pathogen()
            self.pathogen_spawn_timer = self.pathogen_spawn_rate

    def _spawn_pathogen(self):
        angle = self.np_random.uniform(0, 2 * math.pi)
        spawn_x = self.CENTER_X + math.cos(angle) * (self.CELL_RADIUS - 10)
        spawn_y = self.CENTER_Y + math.sin(angle) * (self.CELL_RADIUS - 10)
        
        # Velocity towards center with some randomness
        vel_angle = math.atan2(self.CENTER_Y - spawn_y, self.CENTER_X - spawn_x) + self.np_random.uniform(-0.3, 0.3)
        speed = self.pathogen_base_speed
        
        self.pathogens.append({
            'pos': (spawn_x, spawn_y),
            'vel': (math.cos(vel_angle) * speed, math.sin(vel_angle) * speed),
            'size': self.np_random.integers(6, 9),
            'health': self.pathogen_base_health,
            'active': True,
            'wobble_offset': self.np_random.uniform(0, 2 * math.pi)
        })

    def _deploy_antibody(self):
        if not self.portals: return
        portal_pos = random.choice(self.portals)
        self.antibodies.append({
            'pos': portal_pos,
            'vel': (0,0),
            'size': 5,
            'speed': 2.5,
            'active': True,
            'target': None,
            'lifespan': 300 # 10 seconds at 30fps
        })
        self._create_particles(portal_pos, self.COLOR_ANTIBODY, 15)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': (math.cos(angle) * speed, math.sin(angle) * speed),
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.integers(1, 4)
            })

    def _check_termination(self):
        if self.cell_health <= 0:
            return True
        # Victory condition
        if self.steps > 10 and not self.pathogens:
            return True
        return False

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
            "cell_health": self.cell_health,
            "pathogen_count": len(self.pathogens),
        }

    def _render_game(self):
        # Cell membrane
        pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, self.CELL_RADIUS, self.COLOR_CELL_MEMBRANE)
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, self.CELL_RADIUS, (self.COLOR_CELL_MEMBRANE[0]//2, self.COLOR_CELL_MEMBRANE[1]//2, self.COLOR_CELL_MEMBRANE[2]//2))

        # Nucleus
        glow_color = (*self.COLOR_NUCLEUS, 50)
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, self.NUCLEUS_RADIUS + 5, glow_color)
        pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, self.NUCLEUS_RADIUS, self.COLOR_NUCLEUS)
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, self.NUCLEUS_RADIUS, self.COLOR_NUCLEUS)

        # Portals
        for x, y in self.portals:
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 15, (*self.COLOR_PORTAL, 30))
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 10, self.COLOR_PORTAL)

        # Pathogens
        for p in self.pathogens:
            wobble = math.sin(self.steps * 0.2 + p['wobble_offset']) * 2
            size = int(p['size'] + wobble)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PATHOGEN)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_PATHOGEN)

        # Antibodies
        for a in self.antibodies:
            pos = (int(a['pos'][0]), int(a['pos'][1]))
            s = a['size']
            points = [(pos[0], pos[1] - s), (pos[0] - s, pos[1] + s), (pos[0] + s, pos[1] + s)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ANTIBODY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ANTIBODY)
            
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p['color'], alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0]) - p['size'], int(p['pos'][1]) - p['size']))


        # Internal Walls
        r = self.wall_radii
        cx, cy = self.CENTER_X, self.CENTER_Y
        points = [
            (cx, cy - r['up']), (cx + r['right'], cy),
            (cx, cy + r['down']), (cx - r['left'], cy)
        ]
        for i in range(4):
            pygame.draw.line(self.screen, self.COLOR_WALL, points[i], points[(i + 1) % 4], 4)

    def _render_ui(self):
        # Health Bar
        health_percent = max(0, self.cell_health / 100.0)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, bar_width * health_percent, 20))
        health_text = self.font_ui.render(f"HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Score
        score_text = self.font_title.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Research Progress
        research_pos = (40, self.SCREEN_HEIGHT - 40)
        pygame.gfxdraw.aacircle(self.screen, research_pos[0], research_pos[1], 25, self.COLOR_RESEARCH)
        if self.research_progress > 0:
            end_angle = int(-90 + 360 * (self.research_progress / 100.0))
            pygame.gfxdraw.arc(self.screen, research_pos[0], research_pos[1], 25, -90, end_angle, self.COLOR_RESEARCH)
        research_text = self.font_ui.render("R", True, self.COLOR_TEXT)
        self.screen.blit(research_text, (research_pos[0]-5, research_pos[1]-8))
        
        # Antibody Inventory (simplified)
        inventory_pos = (self.SCREEN_WIDTH - 40, self.SCREEN_HEIGHT - 40)
        pygame.gfxdraw.aacircle(self.screen, inventory_pos[0], inventory_pos[1], 20, self.COLOR_ANTIBODY)
        inv_text = self.font_ui.render(f"x{len(self.antibodies)}", True, self.COLOR_TEXT)
        self.screen.blit(inv_text, (inventory_pos[0] + 25, inventory_pos[1] - 8))
        
    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # This block will not be executed in the testing environment, but is useful for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Create a window to display the game
    pygame.display.set_caption("Cell Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    clock = pygame.time.Clock()
    
    while running:
        action = [0, 0, 0] # Default to no-op
        # Map keyboard inputs to actions for human play
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        clock.tick(30) # Limit frame rate for human play
    
    env.close()