import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:24:26.774489
# Source Brief: brief_00441.md
# Brief Index: 441
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "A stealth puzzle game where you manipulate gravity to navigate a maze. Evade guards and reach the objective by changing which way is down."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press Shift to cycle through gravity cards, and Space to deploy the selected card, changing the direction of gravity."
    )
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 32
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (15, 10, 30)
        self.COLOR_WALL = (40, 30, 60)
        self.COLOR_GRID = (25, 20, 45)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_VISION = (255, 50, 50, 40)
        self.COLOR_OBJECTIVE = (50, 255, 100)
        self.COLOR_ROBOT = (200, 50, 255)
        self.COLOR_PARTICLE = (220, 100, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (25, 20, 45, 180)
        self.COLOR_UI_HIGHLIGHT = (255, 255, 100)

        # --- EXACT spaces: ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_card = pygame.font.SysFont("Consolas", 14)
        self.font_game_over = pygame.font.SysFont("Verdana", 48, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = [0, 0]
        self.objective_pos = [0, 0]
        self.gravity = [0, 1]  # [dx, dy]
        self.walls = []
        self.enemies = []
        self.robots = []
        self.particles = []
        self.card_hand = []
        self.selected_card_idx = None
        self.all_cards = [
            {"name": "GRAV-DOWN", "effect": [0, 1]},
            {"name": "GRAV-UP", "effect": [0, -1]},
            {"name": "GRAV-LEFT", "effect": [-1, 0]},
            {"name": "GRAV-RIGHT", "effect": [1, 0]},
        ]
        
        self.np_random = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Although we use `random`, seeding `np_random` is good practice for reproducibility
            self.np_random = np.random.default_rng(seed=seed)
            random.seed(seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.gravity = [0, 1] # Default gravity: down
        self.robots = []
        self.particles = []
        
        self._generate_level()
        
        # Draw initial hand of 3 cards
        self.card_hand = random.sample(self.all_cards, k=3)
        self.selected_card_idx = 0 if self.card_hand else None

        return self._get_observation(), self._get_info()
    
    def _generate_level(self):
        self.walls = []
        # Create borders
        for x in range(self.GRID_WIDTH):
            self.walls.append(pygame.Rect(x * self.TILE_SIZE, -self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
            self.walls.append(pygame.Rect(x * self.TILE_SIZE, self.HEIGHT, self.TILE_SIZE, self.TILE_SIZE))
        for y in range(self.GRID_HEIGHT):
            self.walls.append(pygame.Rect(-self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
            self.walls.append(pygame.Rect(self.WIDTH, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
        
        # Simple static layout
        layout = [
            "P #         E        ",
            "  # ###############  ",
            "  #               #  ",
            "  # #####         #  ",
            "  #     # #########  ",
            "  #     #         #  ",
            "  ##### #         #  ",
            "      # ###########  ",
            "  #   #           #  ",
            "  #   #########   #  ",
            "  #           #   O  ",
            "  ################## ",
        ]
        for r, row in enumerate(layout):
            for c, char in enumerate(row):
                if char == '#':
                    self.walls.append(pygame.Rect(c * self.TILE_SIZE, r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
                elif char == 'P':
                    self.player_pos = [c, r]
                elif char == 'O':
                    self.objective_pos = [c, r]
                elif char == 'E':
                    self.enemies = [{
                        "pos": [c, r], 
                        "path": [[c, r], [c, r-5], [c+5, r-5], [c+5, r]], 
                        "path_idx": 0, 
                        "speed": 0.05
                    }]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        player_turn_ended = False
        
        prev_dist = self._get_distance_to_objective(self.player_pos)

        # --- Player Action Phase ---
        # Action priority: Deploy > Cycle > Move/Skip
        if space_held and self.selected_card_idx is not None:
            # SFX: Deploy sound effect
            reward += self._deploy_robot()
        elif shift_held:
            # SFX: UI click sound
            self._cycle_card()
        else:
            player_turn_ended = True
            self._move_player(movement)
            new_dist = self._get_distance_to_objective(self.player_pos)
            if new_dist < prev_dist:
                reward += 1.0
            elif new_dist > prev_dist:
                reward -= 1.0

        # --- Enemy & Game State Update Phase ---
        if player_turn_ended:
            self._move_enemies()
            
            if self._check_detection():
                # SFX: Detection alert sound
                self.game_over = True
                reward = -100.0
                self.score += reward
            elif self.player_pos == self.objective_pos:
                # SFX: Victory fanfare
                self.game_over = True
                reward = 100.0
                self.score += reward

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Max steps reached
            reward = -100.0
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_gravity_indicators()
        self._render_walls()
        self._render_objective()
        self._render_robots()
        self._render_enemies()
        self._render_player()
        self._update_and_render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Action Helpers ---
    def _deploy_robot(self):
        if not self.card_hand: return 0.0
        card = self.card_hand.pop(self.selected_card_idx)
        self.gravity = card["effect"]
        self.robots.append({"pos": list(self.player_pos), "type": card["name"]})
        
        # SFX: Gravity shift woosh
        # Spawn particles
        px, py = (self.player_pos[0] + 0.5) * self.TILE_SIZE, (self.player_pos[1] + 0.5) * self.TILE_SIZE
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({"pos": [px, py], "vel": vel, "life": random.randint(20, 40), "radius": random.uniform(2, 5)})
        
        if self.selected_card_idx >= len(self.card_hand):
            self.selected_card_idx = len(self.card_hand) - 1
        if not self.card_hand:
            self.selected_card_idx = None
        return 5.0 # Deployment reward

    def _cycle_card(self):
        if self.card_hand and self.selected_card_idx is not None:
            self.selected_card_idx = (self.selected_card_idx + 1) % len(self.card_hand)

    def _move_player(self, movement):
        if movement == 0: return # No-op
        
        move_map = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]} # Screen-based directions
        screen_move = move_map[movement]
        
        # Translate screen-based move to gravity-based move
        # up=[0,-1], down=[0,1], left=[-1,0], right=[1,0]
        # gravity=[gx,gy]
        # if gravity is down [0,1]: up is [0,-1], down is [0,1], left is [-1,0], right is [1,0] (no change)
        # if gravity is right [1,0]: up is [0,-1], down is [0,1], left is [0,1], right is [0,-1] (rotated)
        # This is a cross product in 2D
        move_vec = [0, 0]
        if self.gravity[0] == 0: # Gravity is Up/Down
            move_vec = [screen_move[0] * self.gravity[1], screen_move[1] * self.gravity[1]]
        else: # Gravity is Left/Right
            move_vec = [screen_move[1] * -self.gravity[0], screen_move[0] * self.gravity[0]]

        target_pos = [self.player_pos[0] + move_vec[0], self.player_pos[1] + move_vec[1]]
        
        # Collision check
        target_rect = pygame.Rect(target_pos[0] * self.TILE_SIZE, target_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        if not any(target_rect.colliderect(wall) for wall in self.walls):
            # SFX: Player step
            self.player_pos = target_pos

    # --- Game Logic Helpers ---
    def _move_enemies(self):
        for enemy in self.enemies:
            path = enemy["path"]
            current_target_idx = (enemy["path_idx"] + 1) % len(path)
            target_pos = path[current_target_idx]
            
            # Move towards target
            direction = [target_pos[0] - enemy["pos"][0], target_pos[1] - enemy["pos"][1]]
            dist = math.hypot(*direction)
            
            if dist < enemy["speed"]:
                enemy["pos"] = list(target_pos)
                enemy["path_idx"] = current_target_idx
            else:
                norm_dir = [d / dist for d in direction]
                enemy["pos"][0] += norm_dir[0] * enemy["speed"]
                enemy["pos"][1] += norm_dir[1] * enemy["speed"]

    def _check_detection(self):
        player_center = ((self.player_pos[0] + 0.5) * self.TILE_SIZE, (self.player_pos[1] + 0.5) * self.TILE_SIZE)
        for enemy in self.enemies:
            cone_points = self._get_vision_cone_points(enemy)
            if self._is_point_in_polygon(player_center, cone_points):
                # Line of sight check
                enemy_center = ((enemy["pos"][0] + 0.5) * self.TILE_SIZE, (enemy["pos"][1] + 0.5) * self.TILE_SIZE)
                has_los = True
                for wall in self.walls:
                    if wall.clipline(enemy_center, player_center):
                        has_los = False
                        break
                if has_los:
                    return True
        return False

    def _get_vision_cone_points(self, enemy):
        enemy_center_px = [(enemy["pos"][0] + 0.5) * self.TILE_SIZE, (enemy["pos"][1] + 0.5) * self.TILE_SIZE]
        
        path = enemy["path"]
        current_target_idx = (enemy["path_idx"] + 1) % len(path)
        target_pos = path[current_target_idx]
        direction = [target_pos[0] - enemy["pos"][0], target_pos[1] - enemy["pos"][1]]
        
        angle = math.atan2(direction[1], direction[0]) if any(d != 0 for d in direction) else 0
        
        vision_range = 5 * self.TILE_SIZE
        vision_angle = math.pi / 4 # 45 degrees total
        
        p1 = enemy_center_px
        p2 = [p1[0] + vision_range * math.cos(angle - vision_angle), p1[1] + vision_range * math.sin(angle - vision_angle)]
        p3 = [p1[0] + vision_range * math.cos(angle + vision_angle), p1[1] + vision_range * math.sin(angle + vision_angle)]
        return [p1, p2, p3]

    def _get_distance_to_objective(self, pos):
        return abs(pos[0] - self.objective_pos[0]) + abs(pos[1] - self.objective_pos[1])

    def _is_point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
        
    # --- Rendering Methods ---
    def _render_background(self):
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

    def _render_player(self):
        px, py = (self.player_pos[0] + 0.5) * self.TILE_SIZE, (self.player_pos[1] + 0.5) * self.TILE_SIZE
        radius = self.TILE_SIZE // 2 - 4
        # Glow effect
        for i in range(radius, radius + 5):
            alpha = self.COLOR_PLAYER_GLOW[3] * ((radius + 5 - i) / 5)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), i, (*self.COLOR_PLAYER_GLOW[:3], int(alpha)))
        pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, self.COLOR_PLAYER)

    def _render_enemies(self):
        for enemy in self.enemies:
            # Vision Cone
            cone_points = self._get_vision_cone_points(enemy)
            pygame.gfxdraw.aapolygon(self.screen, cone_points, self.COLOR_ENEMY_VISION)
            pygame.gfxdraw.filled_polygon(self.screen, cone_points, self.COLOR_ENEMY_VISION)

            # Enemy sprite (triangle)
            ex, ey = (enemy["pos"][0] + 0.5) * self.TILE_SIZE, (enemy["pos"][1] + 0.5) * self.TILE_SIZE
            size = self.TILE_SIZE // 2 - 2
            path = enemy["path"]
            current_target_idx = (enemy["path_idx"] + 1) % len(path)
            target_pos = path[current_target_idx]
            direction = [target_pos[0] - enemy["pos"][0], target_pos[1] - enemy["pos"][1]]
            angle = math.atan2(direction[1], direction[0]) if any(d != 0 for d in direction) else 0

            p1 = (ex + size * math.cos(angle), ey + size * math.sin(angle))
            p2 = (ex + size * math.cos(angle + 2.356), ey + size * math.sin(angle + 2.356)) # 135 deg
            p3 = (ex + size * math.cos(angle - 2.356), ey + size * math.sin(angle - 2.356)) # -135 deg
            points = [p1,p2,p3]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

    def _render_objective(self):
        ox, oy = (self.objective_pos[0] + 0.5) * self.TILE_SIZE, (self.objective_pos[1] + 0.5) * self.TILE_SIZE
        size = self.TILE_SIZE // 2 - 4
        # Pulsating effect
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 5
        rect = pygame.Rect(ox - (size+pulse)/2, oy - (size+pulse)/2, size+pulse, size+pulse)
        pygame.draw.rect(self.screen, self.COLOR_OBJECTIVE, rect, border_radius=3)

    def _render_robots(self):
        for robot in self.robots:
            rx, ry = (robot["pos"][0] + 0.5) * self.TILE_SIZE, (robot["pos"][1] + 0.5) * self.TILE_SIZE
            radius = self.TILE_SIZE // 2 - 6
            pygame.gfxdraw.aacircle(self.screen, int(rx), int(ry), radius, self.COLOR_ROBOT)
            pygame.gfxdraw.filled_circle(self.screen, int(rx), int(ry), radius, self.COLOR_ROBOT)

    def _render_gravity_indicators(self):
        pad = 20
        arrow_len = 30
        arrow_head = 10
        color = (*self.COLOR_ROBOT, 100)
        
        if self.gravity == [0, -1]: # UP
            start, end = (self.WIDTH // 2, pad + arrow_len), (self.WIDTH // 2, pad)
        elif self.gravity == [0, 1]: # DOWN
            start, end = (self.WIDTH // 2, self.HEIGHT - pad - arrow_len), (self.WIDTH // 2, self.HEIGHT - pad)
        elif self.gravity == [-1, 0]: # LEFT
            start, end = (pad + arrow_len, self.HEIGHT // 2), (pad, self.HEIGHT // 2)
        else: # RIGHT
            start, end = (self.WIDTH - pad - arrow_len, self.HEIGHT // 2), (self.WIDTH - pad, self.HEIGHT // 2)
        
        pygame.draw.line(self.screen, color, start, end, 5)
        angle = math.atan2(end[1]-start[1], end[0]-start[0])
        p1 = (end[0] - arrow_head * math.cos(angle - math.pi/6), end[1] - arrow_head * math.sin(angle - math.pi/6))
        p2 = (end[0] - arrow_head * math.cos(angle + math.pi/6), end[1] - arrow_head * math.sin(angle + math.pi/6))
        pygame.draw.line(self.screen, color, end, p1, 5)
        pygame.draw.line(self.screen, color, end, p2, 5)

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = 255 * (p["life"] / 40)
                color = (*self.COLOR_PARTICLE, int(alpha))
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color)

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_ui.render(f"SCORE: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Card Hand
        card_width, card_height = 100, 40
        start_x = (self.WIDTH - len(self.card_hand) * (card_width + 10) + 10) / 2
        for i, card in enumerate(self.card_hand):
            card_rect = pygame.Rect(start_x + i * (card_width + 10), self.HEIGHT - card_height - 10, card_width, card_height)
            
            # Background
            s = pygame.Surface((card_width, card_height), pygame.SRCALPHA)
            s.fill(self.COLOR_UI_BG)
            self.screen.blit(s, card_rect.topleft)
            
            # Border
            border_color = self.COLOR_UI_HIGHLIGHT if i == self.selected_card_idx else self.COLOR_WALL
            pygame.draw.rect(self.screen, border_color, card_rect, 2, border_radius=3)
            
            # Text
            card_name = self.font_card.render(card["name"], True, self.COLOR_UI_TEXT)
            self.screen.blit(card_name, (card_rect.centerx - card_name.get_width() / 2, card_rect.centery - card_name.get_height() / 2))

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        if self.player_pos == self.objective_pos:
            msg = "MISSION COMPLETE"
            color = self.COLOR_OBJECTIVE
        else:
            msg = "AGENT DETECTED"
            color = self.COLOR_ENEMY
            
        text = self.font_game_over.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text, text_rect)

    def render(self):
        return self._get_observation()
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play testing and will not be evaluated.
    # It serves as a good sanity check.
    try:
        env = GameEnv()
        obs, info = env.reset()
        
        # --- Human Play Test ---
        pygame.display.set_caption("Gravity Infiltrator")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        clock = pygame.time.Clock()
        running = True
        
        while running:
            action = [0, 0, 0] # [movement, space, shift]
            
            # Process one action per frame for turn-based feel
            event_processed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and not event_processed:
                    event_processed = True
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP or event.key == pygame.K_w:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        action[0] = 4
                    elif event.key == pygame.K_SPACE:
                        action[1] = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        action[2] = 1
                    else:
                        event_processed = False # No relevant key pressed
            
            if not event_processed:
                # If no key was pressed, we can either do nothing or send a no-op
                # Since this is turn-based, we wait for a key press.
                # To prevent a busy-loop, we just continue.
                # Redrawing the screen is still good.
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                clock.tick(30)
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}")
                # Wait a bit before allowing a reset
                pygame.time.wait(2000)
                obs, info = env.reset()

            # The original code had a tick(10) which is good for human play.
            # Since we now process one action at a time, this is less critical,
            # but still good to have.
            clock.tick(30)

    finally:
        pygame.quit()