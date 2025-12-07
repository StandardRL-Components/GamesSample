import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:03:36.274730
# Source Brief: brief_01453.md
# Brief Index: 1453
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A roguelike platformer Gymnasium environment where the agent climbs a procedurally
    generated clockwork tower. The agent can jump, move, place portals, and rewind time
    to navigate challenges.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Climb a procedurally generated clockwork tower by jumping between platforms, placing portals, and rewinding time."
    user_guide = "Controls: ←→ to move, ↑ to jump. Press space to rewind time and shift to place a portal."
    auto_advance = False

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    MAX_REWIND_HISTORY = 120 # Store 4 seconds of history at 30 FPS
    INITIAL_REWIND_COUNT = 3

    # Physics
    GRAVITY = 0.4
    JUMP_STRENGTH = -9.0
    MOVE_SPEED = 4.0
    FRICTION = 0.85

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_PLATFORM = (0, 100, 150)
    COLOR_PLATFORM_TOP = (0, 150, 200)
    COLOR_OBSTACLE = (200, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 100, 100)
    COLOR_PORTAL_A = (100, 255, 100)
    COLOR_PORTAL_B = (100, 200, 255)
    COLOR_GEAR_INACTIVE = (100, 80, 20)
    COLOR_GEAR_ACTIVE = (255, 215, 0)
    COLOR_TEXT = (220, 220, 220)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # --- State Variables ---
        self.player = {}
        self.platforms = []
        self.obstacles = []
        self.gears = []
        self.particles = []
        self.portals = {}
        self.camera_y = 0
        self.highest_y_generated = 0
        self.level = 0
        self.goal_y = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rewind_charges = 0
        self.state_history = deque(maxlen=self.MAX_REWIND_HISTORY)
        self.next_portal_to_place = 'A'
        self.last_space_held = False
        self.last_shift_held = False

        # --- Background Elements ---
        self._bg_gears = self._generate_bg_gears()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        
        # --- Player ---
        self.player = {
            'x': self.WIDTH / 2, 'y': self.HEIGHT - 50,
            'vx': 0, 'vy': 0,
            'radius': 10, 'on_ground': False
        }

        # --- Camera and World ---
        self.camera_y = 0
        self.highest_y_generated = self.HEIGHT
        self.goal_y = -2000 # Initial goal height
        
        # --- Game Elements ---
        self.platforms = []
        self.obstacles = []
        self.gears = []
        self.particles = []
        self.portals = {'A': None, 'B': None}
        self.next_portal_to_place = 'A'

        # --- Mechanics ---
        self.rewind_charges = self.INITIAL_REWIND_COUNT
        self.state_history.clear()
        self.last_space_held = False
        self.last_shift_held = False

        # --- Level Generation ---
        self._generate_starting_area()
        self._generate_level_chunk(self.HEIGHT - 100)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        
        y_before = self.player['y']
        
        # --- Handle Actions ---
        reward += self._handle_input(action)

        # --- Update Game Logic ---
        self._update_physics()
        self._handle_collisions()
        self._update_camera()
        self._update_particles()
        
        # --- Manage World Generation ---
        if self.player['y'] < self.highest_y_generated - self.HEIGHT * 1.5:
            self._generate_level_chunk(self.highest_y_generated)
            self._prune_offscreen_elements()

        # --- Calculate Rewards ---
        y_after = self.player['y']
        vertical_movement = y_before - y_after
        if vertical_movement > 0:
            reward += 0.1  # Upward movement
        elif vertical_movement < 0:
            reward -= 0.01 # Downward movement

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = False
        if terminated:
            if self.player['y'] > self.camera_y + self.HEIGHT or self._check_obstacle_collision():
                reward = -100 # Fell off or hit obstacle
            elif self.player['y'] <= self.goal_y:
                reward = 100 # Reached goal
                self.level += 1 # For display purposes
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        if movement == 1 and self.player['on_ground']: # Up (Jump)
            self.player['vy'] = self.JUMP_STRENGTH
            self._create_particles(self.player['x'], self.player['y'] + self.player['radius'], count=10, color=(200,200,200))
            # sfx: jump
        # Note: Down (2) is ignored, gravity handles it.
        if movement == 3: # Left
            self.player['vx'] -= self.MOVE_SPEED * 0.2
        if movement == 4: # Right
            self.player['vx'] += self.MOVE_SPEED * 0.2
        
        self.player['vx'] = np.clip(self.player['vx'], -self.MOVE_SPEED, self.MOVE_SPEED)

        # Rewind (Space) - on press
        if space_held and not self.last_space_held and self.rewind_charges > 0:
            if self._use_rewind():
                reward -= 1.0
                self.rewind_charges -= 1
                self._create_particles(self.player['x'], self.player['y'], count=30, color=(150, 150, 255), life=40, speed=4)
                # sfx: rewind
        self.last_space_held = space_held

        # Place Portal (Shift) - on press
        if shift_held and not self.last_shift_held:
            self._place_portal()
            # sfx: portal_place
        self.last_shift_held = shift_held
        
        return reward

    def _update_physics(self):
        # Store state for rewind
        if self.steps % 4 == 0: # Store state every 4 steps
             self._store_state()

        # Apply gravity
        self.player['vy'] += self.GRAVITY
        
        # Apply friction
        self.player['vx'] *= self.FRICTION
        if abs(self.player['vx']) < 0.1: self.player['vx'] = 0

        # Update position
        self.player['x'] += self.player['vx']
        self.player['y'] += self.player['vy']

        # World boundaries (left/right)
        if self.player['x'] - self.player['radius'] < 0:
            self.player['x'] = self.player['radius']
            self.player['vx'] = 0
        if self.player['x'] + self.player['radius'] > self.WIDTH:
            self.player['x'] = self.WIDTH - self.player['radius']
            self.player['vx'] = 0

    def _handle_collisions(self):
        self.player['on_ground'] = False
        player_rect = pygame.Rect(self.player['x'] - self.player['radius'], self.player['y'] - self.player['radius'], self.player['radius']*2, self.player['radius']*2)

        # Platform collisions
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player['vy'] > 0:
                # Check if player was above the platform in the previous frame
                if (self.player['y'] - self.player['vy']) + self.player['radius'] <= plat.top:
                    self.player['y'] = plat.top - self.player['radius']
                    self.player['vy'] = 0
                    self.player['on_ground'] = True
                    # sfx: land
                    
        # Gear collisions
        for gear in self.gears:
            if not gear['activated']:
                dist = math.hypot(self.player['x'] - gear['pos'][0], self.player['y'] - gear['pos'][1])
                if dist < self.player['radius'] + gear['radius']:
                    gear['activated'] = True
                    self._create_particles(gear['pos'][0], gear['pos'][1], count=20, color=self.COLOR_GEAR_ACTIVE, life=30)
                    # sfx: gear_activate
                    
        # Portal collisions
        for p_id, p_dat in self.portals.items():
            if p_dat and not p_dat['cooldown']:
                dist = math.hypot(self.player['x'] - p_dat['pos'][0], self.player['y'] - p_dat['pos'][1])
                if dist < self.player['radius'] + p_dat['radius']:
                    other_id = 'B' if p_id == 'A' else 'A'
                    if self.portals[other_id]:
                        # Teleport
                        self.player['x'], self.player['y'] = self.portals[other_id]['pos']
                        self.portals[other_id]['cooldown'] = self.FPS # 1 second cooldown
                        self.portals[p_id]['cooldown'] = self.FPS
                        self._create_particles(self.portals[p_id]['pos'][0], self.portals[p_id]['pos'][1], count=40, color=p_dat['color'], life=40, speed=5)
                        self._create_particles(self.portals[other_id]['pos'][0], self.portals[other_id]['pos'][1], count=40, color=self.portals[other_id]['color'], life=40, speed=5)
                        # sfx: portal_whoosh
                        break # Exit loop after teleporting
        
        # Update portal cooldowns
        for p_dat in self.portals.values():
            if p_dat and p_dat['cooldown'] > 0:
                p_dat['cooldown'] -= 1

    def _check_termination(self):
        # Fell off bottom
        if self.player['y'] - self.player['radius'] > self.camera_y + self.HEIGHT:
            return True
        # Hit obstacle
        if self._check_obstacle_collision():
            return True
        # Reached goal
        if self.player['y'] <= self.goal_y:
            return True
        return False

    def _check_obstacle_collision(self):
        player_rect = pygame.Rect(self.player['x'] - self.player['radius'], self.player['y'] - self.player['radius'], self.player['radius']*2, self.player['radius']*2)
        for obs in self.obstacles:
            if player_rect.colliderect(obs):
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
            "level": self.level,
            "rewinds": self.rewind_charges,
            "player_y": self.player['y']
        }

    def _render_game(self):
        # --- Background ---
        self._render_bg_gears()

        # --- Game Elements (relative to camera) ---
        cam_x, cam_y = 0, self.camera_y
        
        # Portals
        for p_id, p_dat in self.portals.items():
            if p_dat:
                self._draw_portal(p_dat, cam_y)

        # Gears
        for gear in self.gears:
            self._draw_gear(gear, cam_y)
        
        # Platforms
        for plat in self.platforms:
            p = plat.move(-cam_x, -cam_y)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, (p.x, p.y, p.width, 4), border_radius=3)

        # Obstacles
        for obs in self.obstacles:
            o = obs.move(-cam_x, -cam_y)
            pygame.gfxdraw.box(self.screen, o, self.COLOR_OBSTACLE)
            pygame.gfxdraw.rectangle(self.screen, o, self.COLOR_OBSTACLE_GLOW)

        # Particles
        for p in self.particles:
            pos = (int(p['x'] - cam_x), int(p['y'] - cam_y))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))

        # Player
        player_pos = (int(self.player['x'] - cam_x), int(self.player['y'] - cam_y))
        # Glow
        glow_radius = self.player['radius'] * (1.5 + 0.2 * math.sin(self.steps * 0.1))
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_pos[0] - glow_radius, player_pos[1] - glow_radius))
        # Body
        pygame.gfxdraw.filled_circle(self.screen, player_pos[0], player_pos[1], self.player['radius'], self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos[0], player_pos[1], self.player['radius'], self.COLOR_PLAYER)

    def _render_ui(self):
        rewind_text = self.font_main.render(f"REWIND: {self.rewind_charges}", True, self.COLOR_TEXT)
        self.screen.blit(rewind_text, (10, 10))

        level_text = self.font_main.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

        score_text = self.font_small.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))

    def _update_camera(self):
        target_cam_y = self.player['y'] - self.HEIGHT * 0.6
        # Camera only moves up, creating a floor of death
        self.camera_y = min(self.camera_y, target_cam_y)

    def _generate_starting_area(self):
        # Create a solid floor to start on
        start_floor = pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20)
        self.platforms.append(start_floor)
    
    def _generate_level_chunk(self, y_start):
        # Procedurally generate platforms, gears, and obstacles in a vertical slice
        num_platforms = 15
        y = y_start
        last_x = self.WIDTH / 2
        
        for i in range(num_platforms):
            y -= random.uniform(50, 120) # Vertical distance
            x = last_x + random.uniform(-200, 200)
            x = np.clip(x, 50, self.WIDTH - 50)
            width = random.uniform(80, 150)
            
            self.platforms.append(pygame.Rect(x - width/2, y, width, 15))
            last_x = x

            # Add obstacles with increasing probability
            difficulty = 1.0 - (y / self.goal_y) # 0 to 1
            if random.random() < 0.1 + difficulty * 0.2:
                obs_x = x + random.uniform(-100, 100)
                obs_y = y - random.uniform(20, 40)
                self.obstacles.append(pygame.Rect(obs_x - 10, obs_y - 10, 20, 20))

        self.highest_y_generated = y

    def _prune_offscreen_elements(self):
        cull_y = self.camera_y + self.HEIGHT + 100 # Cull elements 100px below screen
        self.platforms = [p for p in self.platforms if p.bottom > self.camera_y]
        self.obstacles = [o for o in self.obstacles if o.bottom > self.camera_y]
        self.gears = [g for g in self.gears if g['pos'][1] > self.camera_y]

    def _store_state(self):
        state = {
            'player': self.player.copy(),
            'portals': {k: v.copy() if v else None for k, v in self.portals.items()},
            'camera_y': self.camera_y,
        }
        self.state_history.append(state)

    def _use_rewind(self):
        if len(self.state_history) > 1:
            # Pop current state, get previous
            self.state_history.pop() 
            last_state = self.state_history.pop()
            self.player = last_state['player']
            self.portals = last_state['portals']
            self.camera_y = last_state['camera_y']
            return True
        return False

    def _place_portal(self):
        p_id = self.next_portal_to_place
        p_color = self.COLOR_PORTAL_A if p_id == 'A' else self.COLOR_PORTAL_B
        
        self.portals[p_id] = {
            'pos': (self.player['x'], self.player['y']),
            'radius': 15,
            'color': p_color,
            'cooldown': 0
        }
        self._create_particles(self.player['x'], self.player['y'], count=20, color=p_color, life=30)
        self.next_portal_to_place = 'B' if p_id == 'A' else 'A'

    def _create_particles(self, x, y, count=10, color=(255,255,255), life=20, speed=2):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            s = random.uniform(0.5, speed)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * s, 'vy': math.sin(angle) * s,
                'life': life + random.uniform(-5, 5),
                'radius': random.uniform(1, 4),
                'color': color
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['radius'] -= 0.05
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _generate_bg_gears(self):
        gears = []
        for _ in range(15):
            gears.append({
                'x': random.randint(0, self.WIDTH),
                'y': random.randint(0, self.HEIGHT),
                'radius': random.randint(40, 150),
                'speed': random.uniform(-0.2, 0.2),
                'color': (random.randint(25, 40), random.randint(30, 45), random.randint(35, 50)),
                'teeth': random.randint(8, 20)
            })
        return gears

    def _render_bg_gears(self):
        for gear in self._bg_gears:
            angle = (self.steps * gear['speed']) % 360
            self._draw_gear_shape(gear['x'], gear['y'] - (self.camera_y * 0.1) % self.HEIGHT, gear['radius'], gear['teeth'], angle, gear['color'])

    def _draw_gear_shape(self, cx, cy, radius, num_teeth, start_angle, color):
        tooth_angle = 360 / (num_teeth * 2)
        for i in range(num_teeth * 2):
            angle = math.radians(start_angle + i * tooth_angle)
            r = radius if i % 2 == 0 else radius * 0.85
            x1 = cx + r * math.cos(angle)
            y1 = cy + r * math.sin(angle)
            
            angle2 = math.radians(start_angle + (i + 1) * tooth_angle)
            r2 = radius if (i+1) % 2 == 0 else radius * 0.85
            x2 = cx + r2 * math.cos(angle2)
            y2 = cy + r2 * math.sin(angle2)
            
            pygame.gfxdraw.line(self.screen, int(x1), int(y1), int(x2), int(y2), color)
        pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), int(radius * 0.7), color)

    def _draw_gear(self, gear, cam_y):
        pos = (int(gear['pos'][0]), int(gear['pos'][1] - cam_y))
        color = self.COLOR_GEAR_ACTIVE if gear['activated'] else self.COLOR_GEAR_INACTIVE
        self._draw_gear_shape(pos[0], pos[1], gear['radius'], 8, self.steps * 2, color)
        if gear['activated']:
             pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(gear['radius'] * 1.2), (*color, 100))

    def _draw_portal(self, p_dat, cam_y):
        pos = (int(p_dat['pos'][0]), int(p_dat['pos'][1] - cam_y))
        radius = p_dat['radius']
        anim_radius = radius * (0.8 + 0.2 * math.sin(self.steps * 0.2))
        
        # Shimmering effect
        for i in range(5):
            alpha = 50 - i * 10
            r = anim_radius + i * 2
            pygame.gfxdraw.ellipse(self.screen, pos[0], pos[1], int(r*0.7), int(r), (*p_dat['color'], alpha))
        
        # Core
        pygame.gfxdraw.filled_ellipse(self.screen, pos[0], pos[1], int(anim_radius*0.7), int(anim_radius), self.COLOR_BG)
        pygame.gfxdraw.ellipse(self.screen, pos[0], pos[1], int(anim_radius*0.7), int(anim_radius), p_dat['color'])

        if p_dat['cooldown'] > 0:
            # Draw cooldown indicator
            angle = 360 * (p_dat['cooldown'] / self.FPS)
            pygame.gfxdraw.arc(self.screen, pos[0], pos[1], radius + 5, 0, int(angle), (255, 255, 255, 100))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage and Human Play ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Switch to a visible driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Clockwork Tower Climber")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print(GameEnv.user_guide)

    while not (terminated or truncated):
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        # K_DOWN is action 2, but has no effect in this implementation
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a moment before closing
            pygame.time.wait(2000)

        clock.tick(env.FPS)
        
    env.close()